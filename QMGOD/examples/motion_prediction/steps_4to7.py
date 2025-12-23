# -*- coding: utf-8 -*-
"""
步骤 4–7（多目标/多候选通用版）
- 选择阶段支持三种组合：
  A) target=POS, candidate=BND  （BND→POS）
  B) target=NEG, candidate=POS  （POS→NEG）
  C) target=BND, candidate=NEG  （NEG→BND）
- 【保留你之前的“取消 WFD 白化”设定】：
  RBF -> 仅做按行 L2 单位化（不做白化、不做投影）
- ★ 本次补丁：
  1) 一旦检测到【候选集或目标集为空 / RBF特征为空 / 选择结果为空】，立刻整条分支跳过：
     - 直接 return，且返回 {"skipped": True, "stop_rest": True}
     - 表示“跳过当前分支且通知上层停止后续分支”
  2) 在 Step7 前，若训练集只有单一类别（仅 0 或仅 1），也直接跳过该分支，返回 {"skipped": True, "stop_rest": True}
"""

import os
import yaml
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple

from adapters import build_model, build_dataset, set_seed, get_globals
from tarot.utils import set_dataloader_params

try:
    import ot as _pot
except Exception as e:
    raise RuntimeError("需要安装 POT： pip install POT\n" + str(e))

from sklearn import svm


def _to_tensors(batch, device):
    inp = batch['input_dict']
    tensors = [x.to(device, non_blocking=True) for k, x in inp.items() if isinstance(x, torch.Tensor)]
    keys = [k for k, x in inp.items() if isinstance(x, torch.Tensor)]
    tensors.append(keys)
    return tensors

@torch.no_grad()
def _encode(model, tensors_and_keys):
    *tensors, keys = tensors_and_keys
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        return model.encode(tensors, keys)
    x0 = tensors[0]
    return x0.view(x0.size(0), -1)

def _floor_and_renorm(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = torch.clamp(w, min=eps)
    return w / (w.sum() + 1e-12)

@torch.no_grad()
def _median_heuristic_gamma(X: torch.Tensor, Y: torch.Tensor, max_samples: int = 4096):
    Z = torch.cat([X, Y], dim=0)
    if Z.size(0) > max_samples:
        idx = torch.randperm(Z.size(0), device=Z.device)[:max_samples]
        Z = Z[idx]
    D = torch.cdist(Z, Z, p=2)
    med = torch.median(D[D > 0])
    if not torch.isfinite(med) or float(med) <= 0:
        med = torch.tensor(1.0, device=Z.device)
    gamma = 1.0 / (2.0 * (med ** 2) + 1e-12)
    return float(gamma)

@torch.no_grad()
def _rbf_features(X: torch.Tensor, Centers: torch.Tensor, gamma: float, chunk: int = 4096):
    # 安全：若任一为空直接返回空特征，避免 torch.cat 空列表
    if X is None or Centers is None or X.numel() == 0 or Centers.numel() == 0:
        dev = X.device if X is not None else 'cpu'
        dt = X.dtype if X is not None else torch.float32
        return torch.empty((0, 0), dtype=dt, device=dev)
    out = []
    for i in range(0, X.size(0), chunk):
        Xi = X[i:i+chunk]
        Xi2 = (Xi ** 2).sum(dim=1, keepdim=True)
        C2  = (Centers ** 2).sum(dim=1, keepdim=True).T
        dist2 = Xi2 + C2 - 2.0 * (Xi @ Centers.T)
        Ki = torch.exp(-gamma * dist2)
        out.append(Ki)
    if len(out) == 0:
        return torch.empty((0, Centers.size(0)), dtype=X.dtype, device=X.device)
    return torch.cat(out, dim=0)

def _wfd(Gc: torch.Tensor, Gt: torch.Tensor, proj_dim: int, reg: float):
    return None, Gc, Gt

@torch.no_grad()
def _sinkhorn_pot_distance(X, Y, a, b, blur=0.1, p=2, verbose=False):
    C = torch.cdist(X, Y, p=2)
    if p == 2:
        C = C ** 2
    C_np = C.detach().cpu().double().numpy()
    med = np.median(C_np[C_np > 0]) if np.any(C_np > 0) else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    C_np = C_np / med
    a = torch.clamp(a, min=1e-8); b = torch.clamp(b, min=1e-8)
    a = a / (a.sum() + 1e-12); b = b / (b.sum() + 1e-12)
    thr = 1e-8
    Ia = (a > thr).nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
    Ib = (b > thr).nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
    a_np = a[Ia].detach().cpu().double().numpy()
    b_np = b[Ib].detach().cpu().double().numpy()
    a_np = a_np / a_np.sum(); b_np = b_np / b_np.sum()
    C_sub = C_np[np.ix_(Ia, Ib)]
    base_reg = float(blur) ** 2 / (med + 1e-12)
    reg_schedule = [base_reg * s for s in [8.0, 4.0, 2.0, 1.0]]
    last_err = None
    for reg in reg_schedule:
        try:
            gamma, log = _pot.sinkhorn(a_np, b_np, C_sub, reg,
                                       numItermax=20000, stopThr=1e-9,
                                       verbose=False, method="sinkhorn_stabilized", log=True)
            if np.isfinite(gamma).all() and gamma.min() >= 0:
                cost = float((gamma * C_sub).sum())
                return torch.tensor(cost, device=X.device, dtype=X.dtype)
        except Exception as e:
            last_err = e
            continue
    try:
        reg = base_reg * 16.0
        gamma, _ = _pot.sinkhorn(a_np, b_np, C_sub, reg,
                                 numItermax=30000, stopThr=1e-8,
                                 verbose=False, method="sinkhorn_stabilized", log=False)
        cost = float((gamma * C_sub).sum())
        return torch.tensor(cost, device=X.device, dtype=X.dtype)
    except Exception as e:
        if verbose:
            print("[OT] stabilized sinkhorn still failed, returning upper bound; err=", e or last_err)
        cost = float(np.min(C_sub, axis=1).sum())
        return torch.tensor(cost, device=X.device, dtype=X.dtype)

@torch.no_grad()
def _sinkhorn_pot_gamma(X: torch.Tensor, Y: torch.Tensor,
                        a: torch.Tensor, b: torch.Tensor,
                        blur: float = 0.1, p: int = 2,
                        return_potentials: bool = True):
    C = torch.cdist(X, Y, p=2)
    if p == 2:
        C = C ** 2
    C_cpu = C.detach().cpu().double()
    mask_pos = C_cpu > 0
    med = float(torch.median(C_cpu[mask_pos]).item()) if mask_pos.any() else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    C_np = (C_cpu.numpy() / med)
    a = torch.clamp(a, min=1e-6); b = torch.clamp(b, min=1e-6)
    a = a / (a.sum() + 1e-12); b = b / (b.sum() + 1e-12)
    a_np = a.detach().cpu().double().numpy()
    b_np = b.detach().cpu().double().numpy()
    base_reg = float(blur) ** 2 / (med + 1e-12)
    reg_schedule = [base_reg * s for s in (8.0, 4.0, 2.0, 1.0)]
    gamma_np, log, reg_used = None, {}, None
    NUM_ITER = 30000
    STOP_THR = 1e-8
    for reg in reg_schedule:
        try:
            gamma_np, log = _pot.sinkhorn(
                a_np, b_np, C_np, reg,
                numItermax=NUM_ITER, stopThr=STOP_THR,
                verbose=False, method="sinkhorn_stabilized", log=True
            )
            if np.isfinite(gamma_np).all() and gamma_np.min() >= 0:
                reg_used = reg
                break
        except Exception:
            gamma_np = None
    if gamma_np is None:
        try:
            reg_used = base_reg * 16.0
            gamma_np, log = _pot.sinkhorn(
                a_np, b_np, C_np, reg_used,
                numItermax=NUM_ITER, stopThr=STOP_THR,
                verbose=False, method="sinkhorn_stabilized", log=True
            )
        except Exception:
            gamma_np = None
    if gamma_np is None:
        reg_used = base_reg * 4.0
        gamma_np, log = _pot.sinkhorn(
            a_np, b_np, C_np, reg_used,
            numItermax=NUM_ITER, stopThr=STOP_THR,
            verbose=False, method="sinkhorn_log", log=True
        )
    gamma = torch.tensor(gamma_np, device=X.device, dtype=X.dtype)
    if not return_potentials:
        return gamma, None, None
    logu = log.get('logu', None); logv = log.get('logv', None)
    u = log.get('u', None);       v = log.get('v', None)
    if logu is not None:
        f = torch.tensor(reg_used * logu, device=X.device, dtype=X.dtype)
    elif u is not None:
        f = torch.tensor(reg_used * np.log(u + 1e-18), device=X.device, dtype=X.dtype)
    else:
        f = torch.zeros(a.numel(), device=X.device, dtype=X.dtype)
    if logv is not None:
        g = torch.tensor(reg_used * logv, device=X.device, dtype=X.dtype)
    elif v is not None:
        g = torch.tensor(reg_used * np.log(v + 1e-18), device=X.device, dtype=X.dtype)
    else:
        g = torch.zeros(b.numel(), device=X.device, dtype=X.dtype)
    return gamma, f, g

def _otm_select(S: torch.Tensor, Z_cand: torch.Tensor, Z_tgt: torch.Tensor,
                kfold: int = 5, start_k: int = 1, max_k: int = 2000,
                min_ratio: float = 0.02, epsilon: float = 1e-3, device: str = "cpu"):
    Nb, Nt = S.shape
    K = min(max_k, Nb)
    topk_vals, topk_idx = torch.topk(S, k=K, dim=0, largest=True, sorted=True)
    g = torch.Generator(device=S.device).manual_seed(42)
    perm = torch.randperm(Nt, generator=g, device=S.device)
    folds = perm.chunk(kfold)
    chosen = []
    for fold in folds:
        tgt = Z_tgt[fold]
        best = None
        best_set = None
        patience = 0
        for k in range(start_k, K + 1):
            cand_k = topk_idx[:k, fold].reshape(-1)
            uniq, _ = torch.unique(cand_k, return_counts=True)
            a = torch.full((uniq.numel(),), 1.0 / uniq.numel(), device=device)
            b = torch.full((tgt.size(0),), 1.0 / tgt.size(0), device=device)
            dval = float(_sinkhorn_pot_distance(Z_cand[uniq], tgt, a, b, blur=0.1, p=2))
            if best is None or not np.isfinite(best):
                best = dval
                best_set = uniq
                continue
            improve = (best - dval) / max(abs(best), 1e-6)
            if improve > 1e-3:
                best = dval
                best_set = uniq
                patience = 0
            else:
                patience += 1
                if patience >= 2:
                    break
        if best_set is not None:
            chosen.append(best_set)
    if len(chosen) == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, device=device)
    sel_idx = torch.unique(torch.cat(chosen, dim=0))
    min_keep = max(1, int(min_ratio * Nb))
    if sel_idx.numel() < min_keep:
        flat = topk_idx.reshape(-1)
        counts = torch.bincount(flat, minlength=Nb)
        order = torch.argsort(counts, descending=True)
        need = min_keep - sel_idx.numel()
        extra = order[~torch.isin(order, sel_idx)][:need]
        sel_idx = torch.unique(torch.cat([sel_idx, extra], 0))
    flat = topk_idx.reshape(-1)
    weight0 = torch.bincount(flat, minlength=Nb)[sel_idx].float()
    weight0 = weight0 / (weight0.sum() + 1e-12)
    return sel_idx, weight0

def _s_pos(P, mcons, alpha, tau_s_pos):
    a = (mcons - tau_s_pos) / max(1e-12, (1 - tau_s_pos))
    b = (P - alpha)        / max(1e-12, (1 - alpha))
    return torch.minimum(a, b).clamp(0.0, 1.0)

def _s_neg(P, mcons, beta, tau_s_neg):
    a = ((-mcons) - tau_s_neg) / max(1e-12, (1 - tau_s_neg))
    b = (beta - P)             / max(1e-12, beta)
    return torch.minimum(a, b).clamp(0.0, 1.0)

def _reliability_for_target(P_tgt: torch.Tensor, m_tgt: torch.Tensor, target_mode: str,
                            p_pow: float, k_pow: float) -> torch.Tensor:
    target_mode = target_mode.upper()
    if target_mode == "NEG":
        core = (1.0 - P_tgt).clamp(1e-6, 1.0); edge = m_tgt.abs().clamp(1e-6, 1.0)
        r = (core ** p_pow) * (edge ** k_pow)
    elif target_mode == "POS":
        core = (P_tgt).clamp(1e-6, 1.0); edge = m_tgt.abs().clamp(1e-6, 1.0)
        r = (core ** p_pow) * (edge ** k_pow)
    elif target_mode == "BND":
        core = (1.0 - (2.0 * P_tgt - 1.0).abs()).clamp(1e-6, 1.0)
        edge = (1.0 - m_tgt.abs()).clamp(1e-6, 1.0)
        r = (core ** p_pow) * (edge ** k_pow)
    else:
        raise ValueError("target_mode must be one of {'NEG','POS','BND'}")
    return r


def run_selection(cfg_path: str,
                  candidate_mode: str = "BND",
                  target_mode: str = "NEG",
                  run_scoring: bool = True):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device   = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    seed     = int(cfg.get('seed', 42)); set_seed(seed)

    G = get_globals()
    X_all    = G["X_all"];    assert X_all is not None
    IDX_POS  = G["IDX_POS"];  IDX_NEG = G["IDX_NEG"];  IDX_BND = G["IDX_BND"]
    P_all    = torch.tensor(G["P_all"], dtype=torch.float32, device=device)
    m_all    = torch.tensor(G["m_cons_all"], dtype=torch.float32, device=device)
    tau_pos  = float(G["tau_s_pos"]); tau_neg = float(G["tau_s_neg"])

    # ---- 选择阶段（Step4–6） ----
    c_mode = candidate_mode.upper()
    t_mode = target_mode.upper()
    if c_mode == "BND":
        CIDX = IDX_BND; cand_name = "BND"
    elif c_mode == "POS":
        CIDX = IDX_POS; cand_name = "POS"
    elif c_mode == "NEG":
        CIDX = IDX_NEG; cand_name = "NEG"
    else:
        raise ValueError("candidate_mode must be one of {'BND','POS','NEG'}")
    if t_mode == "NEG":
        TIDX = IDX_NEG; tgt_name = "NEG"
    elif t_mode == "POS":
        TIDX = IDX_POS; tgt_name = "POS"
    elif t_mode == "BND":
        TIDX = IDX_BND; tgt_name = "BND"
    else:
        raise ValueError("target_mode must be one of {'NEG','POS','BND'}")

    # ★ 入口前置跳过：候选或目标为空 -> 整条分支跳过，并要求停止后续分支
    if CIDX is None or TIDX is None or len(CIDX) == 0 or len(TIDX) == 0:
        print(f"[run_selection] Skip selector: candidate({cand_name}) or target({tgt_name}) is empty; "
              f"skip this branch (no Step6 & no Step7) and STOP later branches.")
        return {"skipped": True, "stop_rest": True, "reason": "empty candidate or target",
                "selected_idx_rel": torch.empty(0).cpu().numpy()}

    model = build_model(cfg).eval().to(device)
    train_set = build_dataset(cfg)
    beg_set   = build_dataset(cfg, val=True)
    pin_mem = (device == 'cuda')
    bsz_tr  = max(cfg['method']['train_batch_size'] // max(1, len(cfg['devices'])), 1)
    bsz_val = max(cfg['method']['eval_batch_size']  // max(1, len(cfg['devices'])), 1)
    cand_loader = DataLoader(train_set, batch_size=bsz_tr,
                             num_workers=cfg['load_num_workers'], shuffle=False, drop_last=False,
                             collate_fn=train_set.collate_fn, pin_memory=pin_mem)
    beg_loader  = DataLoader(beg_set, batch_size=bsz_val,
                             num_workers=cfg['load_num_workers'], shuffle=False, drop_last=False,
                             collate_fn=train_set.collate_fn, pin_memory=pin_mem)
    cand_loader = set_dataloader_params(cand_loader, 'shuffle', False)
    beg_loader  = set_dataloader_params(beg_loader,  'shuffle', False)

    X_cand = torch.tensor(X_all[CIDX], dtype=torch.float32, device=device)
    X_tgt  = torch.tensor(X_all[TIDX], dtype=torch.float32, device=device)

    # ★ 张量层面的空保护
    if X_cand.numel() == 0 or X_tgt.numel() == 0:
        print(f"[run_selection] Skip selector(after tensor): candidate({cand_name}) or target({tgt_name}) is empty; "
              f"skip this branch and STOP later branches.")
        return {"skipped": True, "stop_rest": True, "reason": "empty candidate or target (tensor)",
                "selected_idx_rel": torch.empty(0).cpu().numpy()}

    center_cap = int(cfg.get('rbf_center_cap', 3000))
    if X_tgt.size(0) > center_cap:
        idx_sub = torch.randperm(X_tgt.size(0), device=device)[:center_cap]
        Centers = X_tgt[idx_sub]
    else:
        Centers = X_tgt

    # ★ Centers 若为空 -> 停止后续
    if Centers is None or Centers.numel() == 0:
        print(f"[run_selection] Skip selector: target({tgt_name}) centers empty; skip this branch and STOP later branches.")
        return {"skipped": True, "stop_rest": True, "reason": "empty centers",
                "selected_idx_rel": torch.empty(0).cpu().numpy()}

    gamma = float(cfg.get('rbf_gamma', 0.0))
    if gamma <= 0:
        gamma = _median_heuristic_gamma(X_cand, X_tgt)

    phi_cand = _rbf_features(X_cand, Centers, gamma=gamma)
    phi_tgt  = _rbf_features(X_tgt,  Centers, gamma=gamma)

    # ★ RBF 特征为空 -> 停止后续
    if phi_cand.numel() == 0 or phi_tgt.numel() == 0 or phi_cand.size(1) == 0 or phi_tgt.size(1) == 0:
        print(f"[run_selection] Skip selector: empty RBF feature; skip this branch and STOP later branches.")
        return {"skipped": True, "stop_rest": True, "reason": "empty rbf feature",
                "selected_idx_rel": torch.empty(0).cpu().numpy()}

    Z_cand = F.normalize(phi_cand, dim=1)
    Z_tgt  = F.normalize(phi_tgt,  dim=1)

    with torch.no_grad():
        S = (Z_cand @ Z_tgt.T).contiguous()

    P_tgt = P_all[TIDX]
    m_tgt = m_all[TIDX]
    p_pow = float(cfg.get('reliability_pow_p', 1.0))
    k_pow = float(cfg.get('reliability_pow_k', 1.0))
    r_tgt = _reliability_for_target(P_tgt, m_tgt.abs(), t_mode, p_pow, k_pow)
    beta_all = _floor_and_renorm(r_tgt)
    a_all    = _floor_and_renorm(torch.full((Z_cand.size(0),), 1.0 / max(Z_cand.size(0), 1), device=device))

    d_ot_all  = _sinkhorn_pot_distance(Z_cand, Z_tgt, a=a_all, b=beta_all,
                                       blur=float(cfg.get('ot_epsilon', 0.1)), p=2)
    print(f"[选择器-{cand_name}→{tgt_name}] 整体 Sinkhorn OT 距离: {float(d_ot_all):.6f}")

    sel_idx_rel, sel_w0 = _otm_select(
        S=S, Z_cand=Z_cand, Z_tgt=Z_tgt,
        kfold=int(cfg.get('kfold', 5)),
        start_k=int(cfg.get('otm_knn_start', 1)),
        max_k=int(cfg.get('otm_max_rank', 2000)),
        min_ratio=float(cfg.get('min_select_ratio', 0.02)),
        epsilon=float(cfg.get('otm_improve_eps', 1e-3)),
        device=device,
    )

    # ★ 选择结果为空 -> 停止后续
    if sel_idx_rel.numel() == 0:
        print(f"[run_selection] Skip selector: selection result is empty; skip this branch and STOP later branches.")
        return {"skipped": True, "stop_rest": True, "reason": "empty selection",
                "selected_idx_rel": torch.empty(0).cpu().numpy()}

    Z_sel = Z_cand[sel_idx_rel]
    a_sel = _floor_and_renorm(torch.full((Z_sel.size(0),), 1.0 / Z_sel.size(0), device=device))
    b_tgt = _floor_and_renorm(beta_all.clone())

    gamma_mat, f_dual, g_dual = _sinkhorn_pot_gamma(
        X=Z_sel, Y=Z_tgt, a=a_sel, b=b_tgt,
        blur=float(cfg.get('ot_epsilon', 0.1)), p=2, return_potentials=True
    )
    w_gamma = gamma_mat.sum(dim=1)
    w_gamma = w_gamma / (w_gamma.sum() + 1e-12)

    if not run_scoring:
        ret = {"weights_gamma": w_gamma.detach().cpu().numpy()}
        if c_mode == "BND":
            ret["selected_bnd_rel"] = sel_idx_rel.detach().cpu().numpy()
        elif c_mode == "POS":
            ret["selected_pos_rel"] = sel_idx_rel.detach().cpu().numpy()
        elif c_mode == "NEG":
            ret["selected_neg_rel"] = sel_idx_rel.detach().cpu().numpy()
        return ret

    # ========= Step7：SVM 打分 =========
    IDX_NEG_now = get_globals()["IDX_NEG"]
    IDX_POS_now = get_globals()["IDX_POS"]

    X_all_np = get_globals()["X_all"]  # (N,D)
    N = X_all_np.shape[0]

    idx_train = np.concatenate([IDX_NEG_now, IDX_POS_now], axis=0).astype(int)
    y_train = np.concatenate([
        np.zeros(len(IDX_NEG_now), dtype=np.int32),
        np.ones(len(IDX_POS_now), dtype=np.int32)
    ], axis=0)

    # ★ 单类/空训练集 -> 停止后续
    if idx_train.size == 0 or np.unique(y_train).size < 2:
        print("[步骤7·SVM] Skip: training set empty or single class; skip this branch Step7 and STOP later branches.")
        return {
            "skipped": True,
            "stop_rest": True,
            "reason": "single-class or empty training set",
            "weights_gamma": w_gamma.detach().cpu().numpy(),
            "selected_idx_rel": sel_idx_rel.detach().cpu().numpy()
        }

    sample_weight = None
    if bool(cfg.get("svm_use_mcons_weight", True)):
        m_all_np = get_globals()["m_cons_all"].astype(np.float32).reshape(-1)
        sample_weight = np.abs(m_all_np[idx_train]).astype(np.float32)

    svm_kernel = str(cfg.get("svm_kernel", "rbf"))
    svm_C      = float(cfg.get("svm_C", 1.0))
    svm_gamma  = cfg.get("svm_gamma", "scale")

    clf = svm.SVC(
        kernel=svm_kernel,
        C=svm_C,
        gamma=svm_gamma,
        class_weight='balanced',
        probability=False
    )
    clf.fit(X_all_np[idx_train], y_train, sample_weight=sample_weight)
    svm_scores = clf.decision_function(X_all_np).astype(np.float32)

    out_dir = cfg.get('output_dir', 'selection_artifacts')
    os.makedirs(out_dir, exist_ok=True)
    suffix = os.environ.get("STEP7_SAVE_SUFFIX", "")
    save_name = f"anomaly_scores_step7{suffix}.pt"
    torch.save({
        'svm_scores': torch.tensor(svm_scores),
        'train_pos': int(len(IDX_POS_now)),
        'train_neg': int(len(IDX_NEG_now)),
        'svm_kernel': svm_kernel,
        'svm_C': svm_C,
        'svm_gamma': svm_gamma,
        'used_mcons_weight': bool(cfg.get("svm_use_mcons_weight", True)),
        'ot_distance_all': float(d_ot_all),
        'rbf_gamma': float(gamma),
        'rbf_centers': int(Centers.size(0)),
        'selected_cand_mode': cand_name,
        'selected_idx_rel': sel_idx_rel.detach().cpu(),
        'weights_gamma': w_gamma.detach().cpu(),
    }, os.path.join(out_dir, save_name))

    print(f"[步骤7·SVM] 已保存 SVM 决策分数到：{os.path.join(out_dir, save_name)} (key='svm_scores')")

    ret = {"weights_gamma": w_gamma.detach().cpu().numpy()}
    if c_mode == "BND":
        ret["selected_bnd_rel"] = sel_idx_rel.detach().cpu().numpy()
    elif c_mode == "POS":
        ret["selected_pos_rel"] = sel_idx_rel.detach().cpu().numpy()
    elif c_mode == "NEG":
        ret["selected_neg_rel"] = sel_idx_rel.detach().cpu().numpy()
    return ret
