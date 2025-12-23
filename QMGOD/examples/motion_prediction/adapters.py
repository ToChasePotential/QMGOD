# adapters.py
# 作用：把 Step1–3 的输出注入给 4–7 用的 build_model / build_dataset / set_seed
#      让 4–7 的 run_selection() “无感知”地跑在 1–3 的结果之上。

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------- 全局缓存：由主程序在跑完 Step1–3 后注入 ----------
_GLOBALS = {
    "device": "cpu",
    # X 全量 (N,D)
    "X_all": None,                 # np.ndarray
    # 三支集合（全局下标）
    "IDX_POS": None,               # np.ndarray[int]
    "IDX_NEG": None,               # np.ndarray[int]  —— 这就是 BEG
    "IDX_BND": None,               # np.ndarray[int]
    # m_cons（全体与 NEG 子集）
    "m_cons_all": None,            # np.ndarray (N,)
    "m_cons_neg": None,            # torch.Tensor (|NEG|,)
    # Step2 融合后的 P(x)（全体）
    "P_all": None,                 # np.ndarray (N,)
    # Step3 自适应后的双端门阈
    "tau_s_pos": None,
    "tau_s_neg": None,
}

# ============== 1) 供主程序注入 1–3 的输出 ==============
def init_from_step13(
    X_all: np.ndarray,
    idx_pos: np.ndarray,
    idx_neg: np.ndarray,
    idx_bnd: np.ndarray,
    m_cons_all: np.ndarray,        # 与 X_all 等长
    P_all: np.ndarray,             # Step2 融合 P(x)
    tau_s_pos: float,
    tau_s_neg: float,
    device: str = None
):
    assert X_all.ndim == 2
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _GLOBALS["device"] = device
    _GLOBALS["X_all"] = X_all.astype(np.float32, copy=False)
    _GLOBALS["IDX_POS"] = idx_pos.astype(np.int64, copy=False)
    _GLOBALS["IDX_NEG"] = idx_neg.astype(np.int64, copy=False)  # = BEG
    _GLOBALS["IDX_BND"] = idx_bnd.astype(np.int64, copy=False)

    # 存全体 m_cons，并派生 NEG 的 m_cons
    _GLOBALS["m_cons_all"] = np.asarray(m_cons_all, dtype=np.float32).reshape(-1)
    _GLOBALS["m_cons_neg"] = torch.tensor(_GLOBALS["m_cons_all"][idx_neg],
                                          dtype=torch.float32, device=device)

    # 融合后的 P(x)
    _GLOBALS["P_all"] = np.asarray(P_all, dtype=np.float32).reshape(-1)

    # 记录自适应后的双端门阈
    _GLOBALS["tau_s_pos"] = float(tau_s_pos)
    _GLOBALS["tau_s_neg"] = float(tau_s_neg)

def get_globals():
    """给 4–7 取全局数组/阈值。"""
    return dict(_GLOBALS)

# 兼容：4–7 要一个“模型”，但我们只做恒等编码
def set_seed(seed: int):
    torch.manual_seed(int(seed))
    # —— GPU: 额外播种 CUDA，保持与 CPU 随机流一致 —— #
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))

class _ArrayDataset(Dataset):
    """把 (N,D) 矩阵包装成 4–7 预期的样式（batch['input_dict'] ...）"""
    def __init__(self, X: np.ndarray, m_cons: torch.Tensor | None):
        # —— 保持在 CPU，由 DataLoader(pin_memory=True) + .to(device, non_blocking=True) 负责搬运 —— #
        self._X = torch.from_numpy(X.astype(np.float32, copy=False)).contiguous()
        self._m = (m_cons.detach().cpu() if isinstance(m_cons, torch.Tensor) else None)
    def __len__(self): return self._X.shape[0]
    def __getitem__(self, i):
        x = self._X[i]       # (D,)
        if self._m is None:
            return {"input_dict": {"x": x}}
        else:
            return {"input_dict": {"x": x, "m_cons": self._m[i]}}
    @staticmethod
    def collate_fn(batch):
        out = {}
        for item in batch:
            for k, v in item["input_dict"].items():
                out.setdefault(k, []).append(v)
        for k in out:
            out[k] = torch.stack(out[k], dim=0)
        return {"input_dict": out}

def build_dataset(cfg: dict, val: bool = False):
    """
    4–7 调用逻辑：
      - BND：build_dataset(cfg)                      -> 训练 split
      - NEG：build_dataset(cfg_neg, val=True)        -> 验证 split
    """
    G = _GLOBALS
    X = G["X_all"]; assert X is not None
    if val:
        # NEG 作为“目标分布”
        idx = G["IDX_NEG"]; m = G["m_cons_neg"]
        return _ArrayDataset(X[idx], m)
    else:
        # BND 作为“候选分布”
        idx = G["IDX_BND"]
        return _ArrayDataset(X[idx], m_cons=None)

class _IdentityEncoder:
    """最小模型：仅提供 encode()，把 batch 里 'x' 原样作为特征返回。"""
    def __init__(self, device="cpu"):
        self._device = device
    def to(self, device): self._device = device; return self
    def eval(self): return self
    def encode(self, tensors, keys):
        x = tensors[0]
        return x.to(self._device)

def build_model(cfg: dict):
    """4–7 只拿模型做 encode，所以返回恒等编码器即可。"""
    return _IdentityEncoder(device=_GLOBALS["device"])
