# -*- coding: utf-8 -*-
import os, yaml, json, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score

from steps_1to3 import (
    step1_build_similarity,
    step2_multiscale_and_fuse_aligned,
    step3_partition_kp_aligned,
)

import adapters
from steps_4to7 import run_selection
import warnings
warnings.filterwarnings("ignore", message="Sinkhorn did not converge", module="ot.bregman._sinkhorn")

# --- 安全加载（兼容 PyTorch 2.6 weights_only=True） ---
def safe_torch_load(path: str):
    try:
        from torch.serialization import add_safe_globals
        from numpy._core import multiarray as _ma
        add_safe_globals([_ma._reconstruct])
    except Exception:
        pass
    return torch.load(path, map_location="cpu", weights_only=False)


def _infer_feat_types(X: np.ndarray, max_cat_card: int = 50, int_is_cat: bool = True):
    types = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.dtype == object:
            types.append('cat'); continue
        if int_is_cat and np.issubdtype(col.dtype, np.integer):
            types.append('cat'); continue
        col_nonan = col[~np.isnan(col)] if np.issubdtype(col.dtype, np.floating) else col
        nunique = len(np.unique(col_nonan))
        types.append('cat' if nunique <= max_cat_card else 'num')
    return types


def _encode_categorical_columns(X: np.ndarray, feat_types):
    X = X.copy()
    for j, t in enumerate(feat_types):
        if t != 'cat': continue
        col = X[:, j]
        if col.dtype != object:
            col = col.astype(object)
        _, inv = np.unique(col, return_inverse=True)
        X[:, j] = inv.astype(np.float32)
    return X.astype(np.float32)


def _to_binary_labels(y_raw: np.ndarray, prefer_minority: bool = True):
    y = np.asarray(y_raw)
    if np.issubdtype(y.dtype, np.number) and set(np.unique(y)) <= {0, 1}:
        return y.astype(np.float32)
    uniq, cnt = np.unique(y, return_counts=True)
    if len(uniq) == 2:
        pos = uniq[np.argmin(cnt)] if prefer_minority else uniq[1]
        return (y == pos).astype(np.float32)
    raise ValueError("labels must be binary or a mapping must be provided.")


def load_mat_any_with_label(path: str, x_key="X", y_key="y",
                            feat_types_key=None, infer_if_missing=True,
                            max_cat_card=50, int_is_cat=True,
                            label_in_last_col=True):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            keys = list(npz.keys())

            def _load_feat_types_from_npz(npz_obj, key):
                v = np.asarray(npz_obj[key])
                if v.dtype == object:
                    return [str(s, 'utf-8', 'ignore').strip() if isinstance(s, (bytes, bytearray)) else str(s).strip()
                            for s in np.squeeze(v).tolist()]
                else:
                    return ["".join(r).strip() for r in v.astype(str)]

            if (x_key in npz) and (y_key in npz):
                X_raw = np.asarray(npz[x_key])
                y_raw = np.asarray(npz[y_key]).reshape(-1)
                ft = None
                if feat_types_key and (feat_types_key in npz):
                    ft = _load_feat_types_from_npz(npz, feat_types_key)
                if ft is None and infer_if_missing:
                    ft = _infer_feat_types(X_raw, max_cat_card, int_is_cat)
                X = _encode_categorical_columns(X_raw, ft)
                y = _to_binary_labels(y_raw)
                return X, ft, y

            if 'trandata' in npz:
                data = np.asarray(npz['trandata'])
                assert data.ndim == 2, "npz['trandata'] 必须为二维数组"
                X_raw, y_raw = data[:, :-1], data[:, -1]
                ft = None
                if feat_types_key and (feat_types_key in npz):
                    ft = _load_feat_types_from_npz(npz, feat_types_key)
                if ft is None and infer_if_missing:
                    ft = _infer_feat_types(X_raw, max_cat_card, int_is_cat)
                X = _encode_categorical_columns(X_raw, ft)
                y = _to_binary_labels(y_raw)
                return X, ft, y

            raise KeyError(f"npz中未找到 {x_key}/{y_key} 或 'trandata'；可用键: {keys}")

    def _load_mat(path):
        try:
            import scipy.io as sio
            return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            import h5py
            f = h5py.File(path, "r")
            class H5Wrapper(dict):
                def __init__(self, h5): self.h5 = h5
                def __contains__(self, k): return k in self.h5
                def __getitem__(self, k):
                    v = self.h5[k]
                    if hasattr(v, "shape"):
                        arr = np.array(v); return arr.T
                    return v
                def keys(self): return list(self.h5.keys())
            return H5Wrapper(f)

    mat = _load_mat(path)
    keys = list(mat.keys()) if hasattr(mat, "keys") else []
    if (x_key in mat) and (y_key in mat):
        X_raw = np.asarray(mat[x_key]); y_raw = np.asarray(mat[y_key]).reshape(-1)
        ft = None
        if feat_types_key and (feat_types_key in mat):
            v = np.asarray(mat[feat_types_key])
            ft = [str(s).strip() for s in np.squeeze(v).tolist()] if v.dtype == object else ["".join(r).strip() for r in v.astype(str)]
        if ft is None and infer_if_missing:
            ft = _infer_feat_types(X_raw, max_cat_card, int_is_cat)
        X = _encode_categorical_columns(X_raw, ft); y = _to_binary_labels(y_raw)
        return X, ft, y
    if 'trandata' in mat:
        data = np.asarray(mat['trandata']); assert data.ndim == 2
        X_raw, y_raw = data[:, :-1], data[:, -1]
        ft = None
        if feat_types_key and (feat_types_key in mat):
            v = np.asarray(mat[feat_types_key])
            ft = [str(s).strip() for s in np.squeeze(v).tolist()] if v.dtype == object else ["".join(r).strip() for r in v.astype(str)]
        if ft is None and infer_if_missing:
            ft = _infer_feat_types(X_raw, max_cat_card, int_is_cat)
        X = _encode_categorical_columns(X_raw, ft); y = _to_binary_labels(y_raw)
        return X, ft, y
    raise KeyError(f"cannot find {x_key}/{y_key} or 'trandata'; keys={keys}")


def pricesafe(x):
    return x


def _diag_partition(tag: str, POS: np.ndarray, NEG: np.ndarray, BND: np.ndarray, y: np.ndarray):
    def stats(idx):
        n = len(idx)
        if n == 0:
            return 0, 0, 0.0
        y1 = int(np.sum(y[idx] == 1))
        y0 = n - y1
        r = y1 / n if n > 0 else 0.0
        return y1, y0, r
    p1, p0, pr = stats(POS)
    n1, n0, nr = stats(NEG)
    b1, b0, br = stats(BND)
    print(
        f"[Step6-{tag}] POS: n={len(POS)} | y1={p1} y0={p0} | ratio={pr:.4f}  ||  "
        f"NEG: n={len(NEG)} | y1={n1} y0={n0} | ratio={nr:.4f}  ||  "
        f"BND: n={len(BND)} | y1={b1} y0={b0} | ratio={br:.4f}"
    )


# ============== S3（MGBOD 路线）：Step3 后直接 SVM 打分 ==============
def eval_branch_S3_like_MGBOD(tag: str, X: np.ndarray, y: np.ndarray,
                              POS: np.ndarray, NEG: np.ndarray, BND: np.ndarray,
                              m_all: np.ndarray, out_base: str, cfg: dict):
    from sklearn import svm
    os.makedirs(out_base, exist_ok=True)

    # NEG-only z-score（对全体样本应用同一变换）
    neg_mean = X[NEG].mean(axis=0)
    neg_std  = X[NEG].std(axis=0, ddof=0)
    neg_std[neg_std < 1e-12] = 1.0
    Xz = (X - neg_mean) / neg_std

    # 训练集与类权
    idx_train = np.concatenate([NEG, POS], axis=0).astype(int)
    y_train = np.concatenate([
        np.zeros(len(NEG), dtype=np.int32),
        np.ones(len(POS), dtype=np.int32)
    ], axis=0)
    t = float(len(POS)) / max(1, (len(POS) + len(NEG)))
    class_weight = None if (t <= 0 or t >= 1) else {0: 1.0 / max(t, 1e-6), 1: 1.0 / max(1.0 - t, 1e-6)}

    use_w = bool(cfg.get("svm_use_mcons_weight_s3", False))  # 默认为 False，更贴 MGBOD
    sample_weight = np.abs(m_all[idx_train]).astype(np.float32) if use_w else None

    clf = svm.SVC(
        kernel=str(cfg.get("svm_kernel_s3", cfg.get("svm_kernel", "rbf"))),
        C=float(cfg.get("svm_C_s3", cfg.get("svm_C", 1.0))),
        gamma=cfg.get("svm_gamma_s3", cfg.get("svm_gamma", "scale")),
        class_weight=class_weight,
        probability=False
    )
    if idx_train.size > 0:
        clf.fit(Xz[idx_train], y_train, sample_weight=sample_weight)
        scores = clf.decision_function(Xz).astype(np.float32)
    else:
        scores = np.zeros(X.shape[0], dtype=np.float32)

    save_path = os.path.join(out_base, f"anomaly_scores_step7__{tag}.pt")
    torch.save({
        'svm_scores': torch.tensor(scores),
        'train_pos': int(len(POS)), 'train_neg': int(len(NEG)),
        'svm_kernel': str(cfg.get("svm_kernel_s3", cfg.get("svm_kernel", "rbf"))),
        'svm_C': float(cfg.get("svm_C_s3", cfg.get("svm_C", 1.0))),
        'svm_gamma': cfg.get("svm_gamma_s3", cfg.get("svm_gamma", "scale")),
        'used_mcons_weight': use_w,
    }, save_path)
    print(f"[步骤7·SVM·S3] 已保存：{save_path} (key='svm_scores')")

    auroc = roc_auc_score(y_true=y.astype(int), y_score=pricesafe(scores))
    auprc = average_precision_score(y_true=y.astype(int), y_score=pricesafe(scores))
    np.save(os.path.join(out_base, f"final_scores__{tag}.npy"), scores)
    print(f"[BRANCH {tag}] AUROC={auroc:.6f}  AUPRC={auprc:.6f}")
    return {"AUROC": float(auroc), "AUPRC": float(auprc), "scores_path": os.path.join(out_base, f"final_scores__{tag}.npy")}


def main(cfg_path: str, data_path: str, out_dir: str = "artifacts_aligned"):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    print(f"[Device] Using device: {device}")

    # ---- 读数据 ----
    X, feat_types, y = load_mat_any_with_label(
        data_path,
        x_key=cfg.get('x_key', 'X'),
        y_key=cfg.get('y_key', 'y'),
        feat_types_key=cfg.get('feat_types_key', None),
        infer_if_missing=True,
        max_cat_card=cfg.get('max_cat_card', 50),
        int_is_cat=cfg.get('int_is_cat', True),
        label_in_last_col=True
    )
    N, _ = X.shape

    # ---- Step1 ----
    R_cache = step1_build_similarity(
        X, feat_types,
        k_sigma=cfg.get('k_sigma', 15),
        tau_sigma=cfg.get('tau_sigma', 1.0),
        q_exp=cfg.get('q_exp', 2.0),
        eta_cat=cfg.get('eta_cat', None),
        lambda_density=cfg.get('lambda_density', 0.5),
        floor_rho=cfg.get('floor_rho', 0.10),
        delta_num=cfg.get('delta_num', 0.8)
    )

    # ---- Step2 ----
    fused = step2_multiscale_and_fuse_aligned(
        X, feat_types, R_cache, y=y,
        max_layers=cfg.get('max_layers', 6),
        weight_gamma=float(cfg.get('mcons_weight_gamma', 0.5)),
        weight_cap=float(cfg.get('mcons_weight_cap', 2.0)),
        js_bins=cfg.get('js_bins', 12),
        js_alpha=cfg.get('js_alpha', 1.0),
        js_pow=cfg.get('js_pow', 1.0),
    )

    # ---- Step3 ----
    P_all = fused['P']; m_all = fused['m_cons']
    POS, NEG, BND, thr_neg, thr_pos = step3_partition_kp_aligned(
        P_all, y, k_prop=float(cfg.get('k_prop', 0.7)), use_label_p=True
    )
    print(f"[Diag] Partition thresholds: NEG_max={thr_neg:.4f} | POS_min={thr_pos:.4f}")
    print(f"[Step3] partition counts: POS={len(POS)}, NEG={len(NEG)}, BND={len(BND)}")
    _diag_partition("S3", POS, NEG, BND, y)

    # ===== StepA–D：OTM 选择（保持不变，但加入跳过控制） =====
    halt_all = False
    allow_6A = True
    allow_6AB = True
    allow_6ABC = True
    allow_6ABCD = True

    POS1 = NEG1 = BND1 = None
    POS2 = NEG2 = BND2 = None
    POS3 = NEG3 = BND3 = None
    POS4 = NEG4 = BND4 = None

    # ---- Step6-A：BND → POS ----
    if not halt_all:
        adapters.init_from_step13(
            X_all=X,
            idx_pos=POS, idx_neg=NEG, idx_bnd=BND,
            m_cons_all=m_all,
            P_all=P_all,
            tau_s_pos=float(cfg.get('tau_m_pos', 0.6)),
            tau_s_neg=float(cfg.get('tau_m_neg', 0.6)),
            device=device,
        )
        retA = run_selection(cfg_path=cfg_path, candidate_mode="BND", target_mode="POS", run_scoring=False)
        if isinstance(retA, dict) and retA.get("skipped", False) and retA.get("stop_rest", False):
            print("[MAIN] Step6-A 触发跳过：停止当前与后续所有分支（含对应 Step7/SVM）。")
            allow_6A = False
            halt_all = True
        else:
            SEL_rel_A = retA.get("selected_bnd_rel", np.array([], dtype=int))
            SEL_abs_A = BND[SEL_rel_A] if SEL_rel_A.size > 0 else np.array([], dtype=int)
            POS1 = np.unique(np.concatenate([POS, SEL_abs_A])) if SEL_rel_A.size > 0 else POS
            BND1 = np.setdiff1d(BND, SEL_abs_A) if SEL_rel_A.size > 0 else BND
            NEG1 = NEG.copy()
            _diag_partition("A", POS1, NEG1, BND1, y)

    # ---- Step6-B：POS → NEG ----
    if not halt_all:
        adapters.init_from_step13(
            X_all=X, idx_pos=POS1, idx_neg=NEG1, idx_bnd=BND1,
            m_cons_all=m_all, P_all=P_all,
            tau_s_pos=float(cfg.get('tau_m_pos', 0.6)),
            tau_s_neg=float(cfg.get('tau_m_neg', 0.6)),
            device=device,
        )
        retB = run_selection(cfg_path=cfg_path, candidate_mode="POS", target_mode="NEG", run_scoring=False)
        if isinstance(retB, dict) and retB.get("skipped", False) and retB.get("stop_rest", False):
            print("[MAIN] Step6-B 触发跳过：停止当前与后续所有分支（含对应 Step7/SVM）。")
            allow_6AB = False
            halt_all = True
        else:
            SEL_rel_B = retB.get("selected_pos_rel", np.array([], dtype=int))
            SEL_abs_B = POS1[SEL_rel_B] if SEL_rel_B.size > 0 else np.array([], dtype=int)
            NEG2 = np.unique(np.concatenate([NEG1, SEL_abs_B])) if SEL_rel_B.size > 0 else NEG1
            POS2 = np.setdiff1d(POS1, SEL_abs_B) if SEL_rel_B.size > 0 else POS1
            BND2 = BND1.copy()
            _diag_partition("B", POS2, NEG2, BND2, y)

    # ---- Step6-C：NEG → BND ----
    if not halt_all:
        adapters.init_from_step13(
            X_all=X, idx_pos=POS2, idx_neg=NEG2, idx_bnd=BND2,
            m_cons_all=m_all, P_all=P_all,
            tau_s_pos=float(cfg.get('tau_m_pos', 0.6)),
            tau_s_neg=float(cfg.get('tau_m_neg', 0.6)),
            device=device,
        )
        retC = run_selection(cfg_path=cfg_path, candidate_mode="NEG", target_mode="BND", run_scoring=False)
        if isinstance(retC, dict) and retC.get("skipped", False) and retC.get("stop_rest", False):
            print("[MAIN] Step6-C 触发跳过：停止当前与后续所有分支（含对应 Step7/SVM）。")
            allow_6ABC = False
            halt_all = True
        else:
            SEL_rel_C = retC.get("selected_neg_rel", np.array([], dtype=int))
            SEL_abs_C = NEG2[SEL_rel_C] if SEL_rel_C.size > 0 else np.array([], dtype=int)
            BND3 = np.unique(np.concatenate([BND2, SEL_abs_C])) if SEL_rel_C.size > 0 else BND2
            NEG3 = np.setdiff1d(NEG2, SEL_abs_C) if SEL_rel_C.size > 0 else NEG2
            POS3 = POS2.copy()
            _diag_partition("C", POS3, NEG3, BND3, y)

    # ---- Step6-D：BND → POS ----
    if not halt_all:
        adapters.init_from_step13(
            X_all=X, idx_pos=POS3, idx_neg=NEG3, idx_bnd=BND3,
            m_cons_all=m_all, P_all=P_all,
            tau_s_pos=float(cfg.get('tau_m_pos', 0.6)),
            tau_s_neg=float(cfg.get('tau_m_neg', 0.6)),
            device=device,
        )
        retD = run_selection(cfg_path=cfg_path, candidate_mode="BND", target_mode="POS", run_scoring=False)
        if isinstance(retD, dict) and retD.get("skipped", False) and retD.get("stop_rest", False):
            print("[MAIN] Step6-D 触发跳过：停止当前与后续所有分支（含对应 Step7/SVM）。")
            allow_6ABCD = False
            halt_all = True
        else:
            SEL_rel_D = retD.get("selected_bnd_rel", np.array([], dtype=int))
            SEL_abs_D = BND3[SEL_rel_D] if SEL_rel_D.size > 0 else np.array([], dtype=int)
            POS4 = np.unique(np.concatenate([POS3, SEL_abs_D])) if SEL_rel_D.size > 0 else POS3
            BND4 = np.setdiff1d(BND3, SEL_abs_D) if SEL_rel_D.size > 0 else BND3
            NEG4 = NEG3.copy()
            _diag_partition("D", POS4, NEG4, BND4, y)

    # ========== 五条分支（完全隔离）==========
    out_base = cfg.get("output_dir", out_dir)
    os.makedirs(out_base, exist_ok=True)

    # --- S3：MGBOD 路线（不走 Step4–6） ---
    results = {}
    results["S3"] = eval_branch_S3_like_MGBOD(
        tag="S3", X=X, y=y, POS=POS, NEG=NEG, BND=BND,
        m_all=m_all, out_base=out_base, cfg=cfg
    )

    # --- 其余四条：沿用 Step4–7 的 SVM，分支互不串扰 ---
    def _eval_branch_and_save_via_step47(tag: str, pos_idx, neg_idx, bnd_idx):
        """
        若 run_selection 返回 {'skipped': True} 或未产生有效分数文件（或长度!=N），
        则跳过该分支评测并返回 {'skipped': True, 'stop_rest': bool}，由上层决定是否终止后续分支。
        """
        adapters.init_from_step13(
            X_all=X, idx_pos=pos_idx, idx_neg=neg_idx, idx_bnd=bnd_idx,
            m_cons_all=m_all, P_all=P_all,
            tau_s_pos=float(cfg.get('tau_m_pos', 0.6)),
            tau_s_neg=float(cfg.get('tau_m_neg', 0.6)),
            device=device,
        )
        os.environ["STEP7_SAVE_SUFFIX"] = f"__{tag}"
        try:
            ret = run_selection(cfg_path=cfg_path, candidate_mode="BND", target_mode="NEG", run_scoring=True)
        finally:
            os.environ.pop("STEP7_SAVE_SUFFIX", None)

        # 1) Step7 主动返回 skipped（例如：训练集为空/单类），直接跳过当前评测，并按需阻断后续
        if isinstance(ret, dict) and ret.get("skipped", False):
            print(f"[BRANCH {tag}] 跳过 Step7/SVM（上游 run_selection 返回 skipped）。")
            return {"skipped": True, "stop_rest": bool(ret.get("stop_rest", False))}

        # 2) 未生成当前分支的产物文件或文件长度与 N 不匹配，同样跳过并阻断后续
        cand = [
            os.path.join(out_base, f"anomaly_scores_step7__{tag}.pt"),
            # 禁止兜底加载其他分支遗留文件，避免长度不一致
        ]
        st = None
        for p in cand:
            if os.path.exists(p):
                st = safe_torch_load(p); break
        if (st is None) or ("svm_scores" not in st):
            print(f"[BRANCH {tag}] 跳过：未找到当前分支的有效 Step7 产物。")
            return {"skipped": True, "stop_rest": True}

        scores = st["svm_scores"].numpy().reshape(-1)
        if scores.shape[0] != N:
            print(f"[BRANCH {tag}] 跳过：scores 长度({scores.shape[0]}) != N({N})。")
            return {"skipped": True, "stop_rest": True}

        print(f"[BRANCH {tag}] 使用 SVM decision_function 作为最终异常分。")
        auroc = roc_auc_score(y_true=y.astype(int), y_score=pricesafe(scores))
        auprc = average_precision_score(y_true=y.astype(int), y_score=pricesafe(scores))
        np.save(os.path.join(out_base, f"final_scores__{tag}.npy"), scores)
        print(f"[BRANCH {tag}] AUROC={auroc:.6f}  AUPRC={auprc:.6f}")
        return {"AUROC": float(auroc), "AUPRC": float(auprc), "scores_path": os.path.join(out_base, f"final_scores__{tag}.npy")}

    # === 根据 allow_* 与是否已计算的分支索引进行严格跳过 ===
    if allow_6A and (POS1 is not None):
        res = _eval_branch_and_save_via_step47("6A", POS1, NEG1, BND1)
        if isinstance(res, dict) and res.get("skipped", False):
            print("[MAIN] 6A 分支被跳过。")
            if res.get("stop_rest", False):
                allow_6AB = allow_6ABC = allow_6ABCD = False
        else:
            results["6A"] = res
    else:
        print("[BRANCH 6A] 跳过 Step7/SVM（因 Step6-A 触发跳过或未完成该分支的选择）。")

    if allow_6AB and (POS2 is not None):
        res = _eval_branch_and_save_via_step47("6AB", POS2, NEG2, BND2)
        if isinstance(res, dict) and res.get("skipped", False):
            print("[MAIN] 6AB 分支被跳过。")
            if res.get("stop_rest", False):
                allow_6ABC = allow_6ABCD = False
        else:
            results["6AB"] = res
    else:
        print("[BRANCH 6AB] 跳过 Step7/SVM（因 Step6-B 触发跳过或未完成该分支的选择）。")

    if allow_6ABC and (POS3 is not None):
        res = _eval_branch_and_save_via_step47("6ABC", POS3, NEG3, BND3)
        if isinstance(res, dict) and res.get("skipped", False):
            print("[MAIN] 6ABC 分支被跳过。")
            if res.get("stop_rest", False):
                allow_6ABCD = False
        else:
            results["6ABC"] = res
    else:
        print("[BRANCH 6ABC] 跳过 Step7/SVM（因 Step6-C 触发跳过或未完成该分支的选择）。")

    if allow_6ABCD and (POS4 is not None):
        res = _eval_branch_and_save_via_step47("6ABCD", POS4, NEG4, BND4)
        if isinstance(res, dict) and res.get("skipped", False):
            print("[MAIN] 6ABCD 分支被跳过。")
        else:
            results["6ABCD"] = res
    else:
        print("[BRANCH 6ABCD] 跳过 Step7/SVM（因 Step6-D 触发跳过或未完成该分支的选择）。")

    # 选最佳分支（按 AUROC）
    best_tag = max(results.keys(), key=lambda k: results[k]["AUROC"])
    best = results[best_tag]
    best_scores = np.load(best["scores_path"])
    np.save(os.path.join(out_base, "final_scores.npy"), best_scores)
    with open(os.path.join(out_base, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"AUROC": best["AUROC"], "AUPRC": best["AUPRC"], "best_branch": best_tag}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_base, "metrics_branches.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[FINAL] best_branch={best_tag} | AUROC={best['AUROC']:.6f} | AUPRC={best['AUPRC']:.6f}")
    print(f"[SAVE] final_scores.npy / metrics.json / metrics_branches.json -> {out_base}")


if __name__ == "__main__":
    main(
        cfg_path=r"G:\OutlierDecetion\QMGOD\examples\motion_prediction\tarot_config.yaml",
        data_path=r"G:\OutlierDecetion\QMGOD\examples\dataset\31_satimage-2.npz",
        out_dir=r"G:\OutlierDecetion\QMGOD\examples\result"
    )
