# -*- coding: utf-8 -*-
"""
Step1–3（向参考论文靠拢，保留 JSD 权；与现有 main 保持完全兼容的函数签名）
- Step1：数值/类别属性相似度采用“阈值-线性”口径；密度一致化用二次差；属性聚合使用 min(t-norm)。
         兼容旧参：k_sigma/tau_sigma/q_exp/eta_cat 会被忽略或映射到 delta_num；
- Step2：L1 为样本层（与 FRS_OD 一致）；L>=2 为球层（与 FRS_OD_GB 一致：中心/半径修正、阈值-线性）。
         层内“ASQ累积”打分与参考一致；层间融合采用 1-meanEntropy 权；JSD 用于“层内属性排序/重要度”。
- Step3：k–p 切分与参考一致；取消逐层“方向校正”和 rank-aware 映射（改回参考的常数兜底方案）。
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

# ------------------------------ 常用工具 ------------------------------

def _minmax_1d(x: np.ndarray) -> Tuple[np.ndarray, float]:
    lo, hi = float(np.min(x)), float(np.max(x))
    span = max(hi - lo, 1e-12)
    return (x - lo) / span, span

def _std_safe(x: np.ndarray) -> float:
    s = float(np.std(x))
    return s if s > 1e-12 else 1e-12

def _cdist_euclid(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2, dtype=np.float64))

# ------------------------------ JSD（保留） ------------------------------

def _quantile_bins(z: np.ndarray, B: int) -> np.ndarray:
    qs = np.linspace(0, 1, B + 1)
    edges = np.quantile(z, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    return edges

def _hist_prob_1d(z: np.ndarray, edges: np.ndarray, alpha: float) -> np.ndarray:
    cnt, _ = np.histogram(z, bins=edges)
    p = (cnt.astype(np.float64) + alpha)
    return p / p.sum()

def _js_div(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    def _kl(a, b):
        a = np.clip(a, 1e-12, 1.0)
        b = np.clip(b, 1e-12, 1.0)
        return float((a * (np.log(a) - np.log(b))).sum())
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def compute_jsd_weights_per_layer(
    X: np.ndarray,
    feat_types: List[str],
    balls: List[np.ndarray],  # 样本索引的列表
    B_bins: int = 12,
    alpha_smooth: float = 1.0,
    u: float = 1.0,
) -> np.ndarray:
    """JSD 仅用于“层内属性排序/重要度”，不直接改变 t-norm（保持与参考一致）。"""
    balls = [np.asarray(b, dtype=np.int64).ravel() for b in balls]
    N, M = X.shape
    w_num = np.zeros(M, dtype=np.float64)

    for a in range(M):
        if feat_types[a] == 'num':
            z, _ = _minmax_1d(X[:, a])  # 与参考一致：先 min-max 再做分布比较
            edges = _quantile_bins(z, B_bins)
            p_glob = _hist_prob_1d(z, edges, alpha_smooth)
            J = 0.0
            for GB in balls:
                z_loc = z[GB]
                p_loc = _hist_prob_1d(z_loc, edges, alpha_smooth)
                J += (len(GB) / N) * _js_div(p_loc, p_glob)
            w_num[a] = max(J, 0.0)
        else:
            vals = X[:, a]
            uniq, cnt = np.unique(vals, return_counts=True)
            p_glob = (cnt.astype(np.float64) + alpha_smooth)
            p_glob /= p_glob.sum()
            J = 0.0
            for GB in balls:
                v = vals[GB]
                cnt_loc = np.array([(v == u0).sum() for u0 in uniq], dtype=np.float64)
                p_loc = (cnt_loc + alpha_smooth) / (cnt_loc.sum() + alpha_smooth * len(uniq))
                J += (len(GB) / N) * _js_div(p_loc, p_glob)
            w_num[a] = max(J, 0.0)
    w = np.power(np.clip(w_num, 0.0, None), u)
    if w.sum() <= 0:
        w[:] = 1.0
    return w / w.sum()

# ------------------------------ Step1：属性相似（阈值-线性）+ 密度一致化（二次差） ------------------------------

def _Ra_numeric_threshold_linear(x_raw: np.ndarray, delta: float) -> torch.Tensor:
    """
    数值属性：先 min-max→欧氏距离 D；阈值 t=std(x')/delta；D>t→相似=0；否则 sim=1-D。
    """
    x, _ = _minmax_1d(x_raw)
    n = x.shape[0]
    D = np.abs(x.reshape(n, 1) - x.reshape(1, n))  # [n,n] on [0,1]
    t = _std_safe(x) / delta
    M = 1.0 - D
    M[D > t] = 0.0
    return torch.tensor(M, dtype=torch.float32)

def _Ra_categorical_hard(vals: np.ndarray) -> torch.Tensor:
    """
    类别属性：严格相等=1，不等=0（取消 eta 平滑，向参考靠拢）。
    """
    v = torch.tensor(vals).view(-1, 1)
    return (v == v.T).float()

def _density_consistency_sq(Ra: torch.Tensor, lam: float) -> torch.Tensor:
    """
    密度一致化：den=mean(Ra,1)；rel=exp(-lam*(den_i-den_j)^2)；Ra<-rel*Ra（向参考靠拢，二次差）。
    """
    with torch.no_grad():
        den = Ra.mean(dim=1, keepdim=True)
    diff2 = (den - den.T) ** 2
    return Ra * torch.exp(-lam * diff2)

def step1_build_similarity(
    X: np.ndarray,
    feat_types: List[str],
    # —— 为了兼容 main 的旧参，这里保留原形参并做映射 ——
    k_sigma: int = 15,              # 忽略
    tau_sigma: float = 1.0,         # 忽略
    q_exp: float = 2.0,             # 忽略
    eta_cat: float = None,          # 忽略
    lambda_density: float = 0.5,
    floor_rho: float = 0.10,
    # —— 兼容：若 cfg 未提供，会走默认 0.8；可在 cfg 里额外给 delta_num 覆写 ——
    **kwargs
) -> Dict[str, object]:
    """
    返回：
      R_list: 每个属性的相似矩阵（阈值-线性 + 二次密度一致化）
      floor_rho: t-norm 时的数值下限
      lambda_density: 供 Step2 球层复用
    """
    # 把 delta_num 从 kwargs 里取出来，默认 0.8；并将 floor_rho 调整为较小稳健值。
    delta_num = float(kwargs.get('delta_num', 0.8))
    # floor_rho 过大会导致退化，这里若用户仍给 0.10，内部下调到 0.01
    floor_rho_eff = min(float(floor_rho), 1e-2)

    X = np.asarray(X)
    M = X.shape[1]
    R_list: List[torch.Tensor] = []
    for a in range(M):
        if feat_types[a] == 'num':
            Ra = _Ra_numeric_threshold_linear(X[:, a], delta=delta_num)
        else:
            Ra = _Ra_categorical_hard(X[:, a])
        Ra = _density_consistency_sq(Ra, lam=lambda_density)
        Ra = torch.clamp(Ra, min=float(floor_rho_eff))  # 少量数值保底
        R_list.append(Ra)

    return dict(R_list=R_list,
                floor_rho=float(floor_rho_eff),
                lambda_density=float(lambda_density),
                delta_num=float(delta_num))

# ------------------------------ Step2：GB多层与层内打分（与参考一致：min t-norm + ASQ） ------------------------------

class _GB:
    def __init__(self, X: np.ndarray, M: np.ndarray):
        self.X, self.M = X, M
        self.n = X.shape[0]
        ij = np.argmax(M)
        self.p1, self.p2 = divmod(int(ij), self.n)
        self.c = X.mean(axis=0)
        self.r = float(np.linalg.norm(X - self.c[None, :], axis=1).max())   # ★最大离心半径（向参考靠拢）

    def DM(self) -> float:
        return float(np.linalg.norm(self.X - self.c[None, :], axis=1).mean())

    def split_1(self):
        if self.r <= 1e-3:
            return False, None, None
        d1, d2 = self.M[:, self.p1], self.M[:, self.p2]
        m1 = d1 < d2
        m2 = ~m1
        if m1.sum() == 0 or m2.sum() == 0:
            return False, None, None
        X1, X2 = self.X[m1], self.X[m2]
        M1, M2 = self.M[m1][:, m1], self.M[m2][:, m2]
        if M1.size == 0 or M2.size == 0 or np.max(M1) == 0 or np.max(M2) == 0:
            return False, None, None
        g1, g2 = _GB(X1, M1), _GB(X2, M2)
        DM, DM1, DM2 = self.DM(), g1.DM(), g2.DM()
        wDM = (DM1 * X1.shape[0] + DM2 * X2.shape[0]) / float(self.X.shape[0])
        return (wDM < DM), g1, g2

    def split_2(self):
        d1, d2 = self.M[:, self.p1], self.M[:, self.p2]
        m1 = d1 < d2
        m2 = ~m1
        X1, X2 = self.X[m1], self.X[m2]
        M1, M2 = self.M[m1][:, m1], self.M[m2][:, m2]
        return _GB(X1, M1), _GB(X2, M2)

    def center_radius(self):
        return self.c.astype(np.float64), float(self.r)

def _general_GB_numpy(X: np.ndarray, M: Optional[np.ndarray] = None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if M is None:
        M = _cdist_euclid(X, X)
    gb0 = _GB(X, M)
    stack = [(np.arange(X.shape[0]), gb0)]
    leaves = []
    while stack:
        loc, gb = stack.pop()
        flag, g1, g2 = gb.split_1()
        if flag:
            m1 = gb.M[:, gb.p1] < gb.M[:, gb.p2]
            m2 = ~m1
            stack.append((loc[m1], g1))
            stack.append((loc[m2], g2))
        else:
            leaves.append((loc, gb))
    radii = np.array([gb.r for _, gb in leaves], dtype=np.float64)
    thr = max(float(np.mean(radii)), float(np.median(radii)))
    GB_loc_list, centers, rs = [], [], []
    for (loc, gb), r in zip(leaves, radii):
        if r >= thr and len(loc) >= 2:
            g1, g2 = gb.split_2()
            m1 = gb.M[:, gb.p1] < gb.M[:, gb.p2]
            m2 = ~m1
            loc1, loc2 = loc[m1], loc[m2]
            GB_loc_list.extend([loc1, loc2])
            c1, r1 = g1.center_radius()
            c2, r2 = g2.center_radius()
            centers.extend([c1, c2])
            rs.extend([r1, r2])
        else:
            GB_loc_list.append(loc)
            c0, r0 = gb.center_radius()
            centers.append(c0)
            rs.append(r0)
    centers = np.asarray(centers, dtype=np.float64)
    rs = np.asarray(rs, dtype=np.float64)
    return GB_loc_list, centers, rs

def _get_newM_numpy(C: np.ndarray, r: np.ndarray) -> np.ndarray:
    n = C.shape[0]
    M = _cdist_euclid(C, C)
    np.fill_diagonal(M, 0.0)
    rr = r.reshape(n, 1)
    d_r = _cdist_euclid(rr, -rr)
    np.fill_diagonal(d_r, 0.0)
    M = M - d_r
    M[M < 0] = 0.0
    return M

def _t_norm_min_aggregate(R_list: List[torch.Tensor], subset: List[int]) -> torch.Tensor:
    Rb = None
    for a in subset:
        Rb = R_list[a] if Rb is None else torch.minimum(Rb, R_list[a])  # ★与参考一致（min）
    return Rb

def _sig_of_matrix(M: torch.Tensor, N: int) -> float:
    # 参考：eq_class = M.sum(1)；entropy = log(eq_class/N)；sig = -mean(entropy)
    eq_class = M.sum(dim=1)
    ent = torch.log(torch.clamp(eq_class / float(N), min=1e-12))
    return float(-torch.mean(ent).item())

def _score_via_asq(R_list: List[torch.Tensor], order_attrs: List[int]) -> torch.Tensor:
    """
    与参考 get_score() 对齐：
    score = 1 - (1/|ASQ|) * Σ_i [ mean(M_i,1) * sig(M_i) ]
    其中 M_i 为前 i 个属性的 min-聚合矩阵。
    """
    N = R_list[0].size(0)
    S = torch.ones(N, dtype=torch.float32)
    for i in range(len(order_attrs)):
        subset = order_attrs[: i + 1]
        M_i = _t_norm_min_aggregate(R_list, subset)
        w_i = _sig_of_matrix(M_i, N)
        S = S + M_i.mean(dim=1) * w_i
    S = 1.0 - S / max(1, len(order_attrs))
    return torch.clamp(S, 0.0, 1.0)

def _score_layer_sample(X: np.ndarray, feat_types: List[str], R_cache: Dict[str, object],
                        order_attrs: List[int]) -> np.ndarray:
    R_list = R_cache['R_list']
    S = _score_via_asq(R_list, order_attrs)
    return S.numpy()

def _score_layer_ball(
    X: np.ndarray, feat_types: List[str], R_cache: Dict[str, object],
    balls_idx: List[np.ndarray], C: np.ndarray, r: np.ndarray,
    order_attrs: List[int], delta_num: float
) -> np.ndarray:
    """
    按参考 FRS_OD_GB.get_matrix 的思路：
    - 每个数值属性单独 min-max；
    - 中心值对的距离减去“半径修正”（r^{1/m}/m）；
    - 阈值-线性（std/δ）得到球-球相似；
    - 多属性用 min 聚合；再按 ASQ 累积；最后回写到样本。
    """
    G, M = C.shape[0], X.shape[1]
    N = X.shape[0]
    R_ball_attrs: List[torch.Tensor] = []

    for a in range(M):
        if feat_types[a] == 'num':
            z_c, span = _minmax_1d(C[:, a])
            n = G
            D = np.abs(z_c.reshape(n, 1) - z_c.reshape(1, n))
            r_scaled = r / max(span, 1e-12)
            r_eff = np.power(np.clip(r_scaled, 0.0, None), 1.0 / max(M, 1)) / max(M, 1)
            d_r = _cdist_euclid(r_eff.reshape(-1, 1), -r_eff.reshape(-1, 1))
            np.fill_diagonal(d_r, 0.0)
            D = D - d_r
            D[D < 0] = 0.0
            t = _std_safe(z_c) / delta_num
            M_ij = 1.0 - D
            M_ij[D > t] = 0.0
            Ra = torch.tensor(M_ij, dtype=torch.float32)
        else:
            vals = np.asarray(X[:, a])
            vc = np.array([np.argmax(np.bincount(vals[idx].astype(int))) for idx in balls_idx])
            v = torch.tensor(vc, dtype=torch.int64).view(-1, 1)
            Ra = (v == v.T).float()

        # 密度一致化（二次差）
        with torch.no_grad():
            den = Ra.mean(dim=1, keepdim=True)
        diff2 = (den - den.T) ** 2
        Ra = Ra * torch.exp(-float(R_cache.get('lambda_density', 0.5)) * diff2)
        Ra = torch.clamp(Ra, min=float(R_cache['floor_rho']))
        R_ball_attrs.append(Ra)

    S_ball = _score_via_asq([R_ball_attrs[a] for a in range(M)], order_attrs)  # (G,)

    belong = np.empty(N, dtype=np.int64)
    for gid, idx in enumerate(balls_idx):
        belong[idx] = gid
    Sx = S_ball[torch.tensor(belong, dtype=torch.long)].numpy()
    return Sx

def _map_join_like_reference(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    与参考 join() 一致：
      - 以 y 的正负样本数划分上下半区；
      - 下半区 minmax→[0,0.5]，若退化用 0.25；
      - 上半区 minmax→[0.5,1]，若退化用 0.75。
    """
    N = len(S)
    S = np.asarray(S, dtype=np.float64).copy()
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = N - n_pos
    order = np.argsort(S)
    lo = order[:n_neg]
    hi = order[-n_pos:] if n_pos > 0 else np.array([], dtype=int)

    if n_neg > 0:
        part = S[lo]
        pmin, pmax = part.min(), part.max()
        if pmax > pmin:
            S[lo] = (part - pmin) / (pmax - pmin) / 2.0
        else:
            S[lo] = 0.25
    if n_pos > 0:
        part = S[hi]
        pmin, pmax = part.min(), part.max()
        if pmax > pmin:
            S[hi] = (part - pmin) / (pmax - pmin) / 2.0 + 0.5
        else:
            S[hi] = 0.75
    return np.clip(S, 0.0, 1.0)

def _entropy_bernoulli(p: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(p + eps) * np.log2(p + eps) - (1 - p + eps) * np.log2(1 - p + eps)

def step2_multiscale_and_fuse_aligned(
    X: np.ndarray,
    feat_types: List[str],
    R_cache: Dict[str, object],
    y: np.ndarray,
    max_layers: int = 6,
    weight_gamma: float = 0.5,
    weight_cap: float = 2.0,
    # —— 兼容 main：其余可选参数允许传入但不强制使用 ——
    **kwargs
) -> Dict[str, object]:

    # 从 R_cache 取 delta/lam 等（step1 已放进去）
    delta_num = float(R_cache.get('delta_num', 0.8))

    X = np.asarray(X, dtype=np.float64)
    N, M = X.shape
    y = np.asarray(y).astype(int)

    Pk_list: List[np.ndarray] = []
    Hk_list: List[np.ndarray] = []
    layers_info = []

    # L1（样本层）—— 用“单样本球”计算该层 JSD 排序
    balls_L1 = [np.array([i], dtype=np.int64) for i in range(N)]
    w_jsd_L1 = compute_jsd_weights_per_layer(X, feat_types, balls_L1,
                                             B_bins=int(kwargs.get('js_bins', 12)),
                                             alpha_smooth=float(kwargs.get('js_alpha', 1.0)),
                                             u=float(kwargs.get('js_pow', 1.0)))
    order_L1 = list(np.argsort(-w_jsd_L1))  # JSD 大→小排序
    print(f"[Step2] L1(sample) #balls = {N}")

    S1 = _score_layer_sample(X, feat_types, R_cache, order_L1)
    P1 = _map_join_like_reference(S1, y)
    H1 = _entropy_bernoulli(P1)
    Pk_list.append(P1); Hk_list.append(H1)
    layers_info.append(dict(type='sample', size_stats=(1, 1.0, 1.0, 1),
                            jsw=w_jsd_L1.astype(np.float32)))

    # L>=2（球层）
    atoms = [np.array([i], dtype=np.int64) for i in range(N)]
    X_cur, M_cur, lid = X.copy(), None, 2
    while lid <= max_layers:
        GB_loc, C, r = _general_GB_numpy(X_cur, M_cur)
        if len(GB_loc) == 0:
            break

        balls_idx = []
        for loc in GB_loc:
            merged = np.concatenate([atoms[u] for u in loc], axis=0)
            balls_idx.append(np.unique(merged))

        sizes = np.array([len(idx) for idx in balls_idx], dtype=int)
        size_stats = (int(sizes.min()), float(np.median(sizes)),
                      float(sizes.mean()), int(sizes.max()))
        print(f"[Step2] L{lid}(ball)  #balls = {len(balls_idx)}   size[min/med/mean/max]={size_stats}")

        # JSD 权（按该层球划分）
        w_jsd = compute_jsd_weights_per_layer(X, feat_types, balls_idx,
                                              B_bins=int(kwargs.get('js_bins', 12)),
                                              alpha_smooth=float(kwargs.get('js_alpha', 1.0)),
                                              u=float(kwargs.get('js_pow', 1.0)))
        order_attrs = list(np.argsort(-w_jsd))

        S_k = _score_layer_ball(X, feat_types, R_cache,
                                balls_idx, C, r,
                                order_attrs, delta_num=delta_num)
        Pk  = _map_join_like_reference(S_k, y)
        Hk  = _entropy_bernoulli(Pk)
        Pk_list.append(Pk); Hk_list.append(Hk)
        layers_info.append(dict(type='ball', GB=balls_idx, centers=C.astype(np.float32),
                                radii=r.astype(np.float32), size_stats=size_stats,
                                jsw=w_jsd.astype(np.float32)))

        # 下一层
        X_next, M_next = C.astype(np.float64), _get_newM_numpy(C.astype(np.float64), r.astype(np.float64))
        atoms = [np.copy(idx) for idx in balls_idx]
        if M_next.size == 0 or np.max(M_next) == 0.0:
            break
        X_cur, M_cur, lid = X_next, M_next, lid + 1

    # 层权（1-meanEntropy)^gamma，cap，再归一
    nus = np.array([1.0 - hk.mean() for hk in Hk_list], dtype=np.float64)
    w = np.power(np.clip(nus, 1e-12, None), float(weight_gamma))
    w = np.minimum(w, w.mean() * float(weight_cap))
    w = w / w.sum()
    print("[Step2] layer weights (1-meanEntropy)^gamma then capped -> normalized:")
    print(f"        K={len(w)}, w=", np.round(w, 4).tolist())

    P_stack = np.stack(Pk_list, axis=1)    # (N,K)
    H_stack = np.stack(Hk_list, axis=1)    # (N,K)
    P_fused = (P_stack * w[None, :]).sum(axis=1)
    H_fused = (H_stack * w[None, :]).sum(axis=1)
    m_cons  = (2.0 * P_fused - 1.0) * (1.0 - H_fused)

    # 分布诊断
    def _hist_str(vals, lo, hi, bins=10):
        hist, edges = np.histogram(vals, bins=np.linspace(lo, hi, bins + 1))
        return ", ".join([f"[{edges[i]:.4f},{edges[i+1]:.4f}):{hist[i]}" for i in range(len(hist))])

    for k in range(P_stack.shape[1]):
        pk = P_stack[:, k]
        qs = np.quantile(pk, np.linspace(0, 1, 11))
        print(f"[Diag] Layer-{k+1} Pk range=[{pk.min():.4f},{pk.max():.4f}] mean={pk.mean():.4f} std={pk.std():.4f}")
        print("       deciles:", [round(float(q), 4) for q in qs])
        print("       hist   :", _hist_str(pk, 0.0, 1.0, 10))

    qs = np.quantile(P_fused, np.linspace(0, 1, 11))
    print(f"[Diag] P(fused) range=[{P_fused.min():.4f}, {P_fused.max():.4f}]  mean={P_fused.mean():.4f}  std={P_fused.std():.4f}")
    print("[Diag] P deciles:", [round(float(q), 4) for q in qs])
    print("[Diag] P hist   :", _hist_str(P_fused, 0.0, 1.0, 10))

    return dict(
        P=P_fused.astype(np.float32),
        m_cons=m_cons.astype(np.float32),
        Hk_list=[torch.tensor(hk, dtype=torch.float32) for hk in Hk_list],
        nu=w.astype(np.float32),
        layers=layers_info,
        lambda_density=float(R_cache.get('lambda_density', 0.5)),
        delta_num=float(delta_num),
    )

# ------------------------------ Step3：k–p 切分（与参考一致） ------------------------------

def step3_partition_kp_aligned(
    P: np.ndarray,
    y: Optional[np.ndarray],
    k_prop: float = 0.7,
    use_label_p: bool = True
):
    N = len(P)
    P = np.asarray(P, dtype=np.float64)
    if y is not None and use_label_p:
        p = float(np.asarray(y).astype(int).sum()) / float(N)
    else:
        p = float((P > np.median(P)).sum()) / float(N)

    n_il = int(np.floor(N * float(k_prop) * (1.0 - p)))
    n_ol = int(np.floor(N * float(k_prop) * p))

    order = np.argsort(P)
    NEG = order[:n_il]
    POS = order[-n_ol:] if n_ol > 0 else np.array([], dtype=int)
    used = np.union1d(NEG, POS)
    BND = np.setdiff1d(np.arange(N), used, assume_unique=False)

    thr_neg = float(P[NEG].max()) if len(NEG) > 0 else float('nan')
    thr_pos = float(P[POS].min()) if len(POS) > 0 else float('nan')
    return POS, NEG, BND, thr_neg, thr_pos