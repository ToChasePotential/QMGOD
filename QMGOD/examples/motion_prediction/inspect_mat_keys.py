import numpy as np

def inspect_mat_keys(path):
    """
    打印 .mat 中的键名、形状、dtype，并自动猜测哪个是特征 X、哪个是标签 y。
    """
    # 1) 读取
    getter = None
    try:
        import scipy.io as sio
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        getter = lambda k: mat[k]
        backend = "scipy.io.loadmat"
    except NotImplementedError:
        import h5py
        f = h5py.File(path, "r")
        keys = list(f.keys())
        # 注意：MATLAB v7.3 维度与 NumPy 相反，这里只做展示，不强行转置
        getter = lambda k: np.array(f[k])
        backend = "h5py (v7.3)"

    print(f"[loader] {backend}")
    print("keys:", keys)

    # 2) 打印每个键的基本信息
    arrays = {}
    for k in keys:
        try:
            arr = np.asarray(getter(k))
            arrays[k] = arr
            shape = arr.shape
            print(f"{k:20s} shape={shape} dtype={arr.dtype}")
        except Exception as e:
            print(f"{k:20s} <non-array> ({type(getter(k))})")

    # 3) 粗略猜测：最大的 2D 数组（列数>1）当作 X
    X_key = None
    max_elems = -1
    for k, a in arrays.items():
        if a.ndim == 2 and a.shape[1] > 1:
            elems = a.size
            if elems > max_elems:
                max_elems = elems
                X_key = k

    # 4) 如果有 X_key，再猜测 y_key（长度与样本数匹配、且可能是二分类）
    y_key = None
    if X_key is not None:
        X = arrays[X_key]
        n = X.shape[0] if X.shape[0] >= X.shape[1] else X.shape[1]  # 兼容转置
        candidates = []
        for k, a in arrays.items():
            flat = a.reshape(-1)
            if flat.shape[0] == n:
                uniq = np.unique(flat)
                candidates.append((k, len(uniq)))
        candidates.sort(key=lambda x: x[1])  # 唯一值最少的更可能是标签
        if candidates:
            y_key = candidates[0][k := 0] if isinstance(candidates[0], tuple) else candidates[0]
            y_key = candidates[0][0]  # 修正

    print("\n[Guess]")
    print("  X_key (features) :", X_key)
    print("  y_key (labels?)  :", y_key)

    # 5) 如果猜到了 y_key，看看取值大致情况
    if y_key is not None:
        y = arrays[y_key].reshape(-1)
        uniq, cnt = np.unique(y, return_counts=True)
        print("  y unique values :", uniq)
        print("  y counts        :", cnt)

# ====== 调用示例 ======
inspect_mat_keys(r"G:\OutlierDecetion\QMGOD\examples\dataset\mammography.mat")
