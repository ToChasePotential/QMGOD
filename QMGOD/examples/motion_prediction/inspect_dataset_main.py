# -*- coding: utf-8 -*-
"""
inspect_dataset_main.py
-----------------------
直接运行这个脚本即可查看一个数据集的基本信息。

使用方法：
1. 在下面 main() 里，把 file_path 改成你的 .npz 或 .mat 数据集的绝对路径。
2. 在 PyCharm 中右键此文件 → Run 'inspect_dataset_main'，查看控制台输出。
"""

import os
import numpy as np

# 如果你的环境里没有 scipy、又需要读 .mat，可以先装：
# conda install scipy
try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def load_npz(path):
    """从 .npz 中尽量自动找出 X (特征) 和 y (标签)。"""
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    # 1) 常见命名
    if "X" in keys and "y" in keys:
        X = data["X"]
        y = data["y"]
        return X, y

    if "data" in keys and "label" in keys:
        X = data["data"]
        y = data["label"]
        return X, y

    # 2) 自动在所有数组中寻找 2D 特征和 1D 标签
    X_candidates = []
    y_candidates = []

    for k in keys:
        arr = data[k]
        arr = np.array(arr)

        if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 1:
            X_candidates.append(arr)
        elif arr.ndim == 1:
            y_candidates.append(arr)
        elif arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
            y_candidates.append(arr.reshape(-1))

    # 尝试匹配：样本数一致
    for X in X_candidates:
        n = X.shape[0]
        for y in y_candidates:
            if y.shape[0] == n:
                return X, y

    # 3) 退一步：如果只有一个 2D 数组，默认最后一列为标签
    if len(X_candidates) == 1:
        X_all = X_candidates[0]
        if X_all.shape[1] >= 2:
            X = X_all[:, :-1]
            y = X_all[:, -1]
            return X, y

    raise ValueError("无法在 npz 文件中自动识别 X 和 y，请检查数据结构。")


def load_mat_generic(path):
    """从 .mat 中尽量自动找出 X 和 y。"""
    if not SCIPY_AVAILABLE:
        raise ImportError("需要 scipy 才能读取 .mat 文件，请先在当前环境中安装 scipy。")

    m = loadmat(path)
    valid_items = {k: v for k, v in m.items() if not k.startswith("__")}

    X_candidates = []
    y_candidates = []

    for _, arr in valid_items.items():
        arr = np.array(arr)

        if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 1:
            if arr.shape[0] > 1 and arr.shape[1] > 1:
                X_candidates.append(arr)
            else:
                y_candidates.append(arr.reshape(-1))
        elif arr.ndim == 1:
            y_candidates.append(arr)

    # 匹配样本数一致的 X、y
    for X in X_candidates:
        n = X.shape[0]
        for y in y_candidates:
            if y.shape[0] == n:
                return X, y

    # 退一步：只有一个二维数组时，最后一列当标签
    if len(X_candidates) == 1:
        X_all = X_candidates[0]
        if X_all.shape[1] >= 2:
            X = X_all[:, :-1]
            y = X_all[:, -1]
            return X, y

    raise ValueError("无法在 mat 文件中自动识别 X 和 y，请检查数据结构。")


def infer_feature_types(X, max_cat_unique=20, cat_ratio=0.05):
    """
    简单判断每一列是数值属性还是类别属性：
    - 非数字类型 -> 类别属性
    - 数值类型：
        如果去重后唯一值个数 <= min(max_cat_unique, cat_ratio * n_samples)，认为是类别属性
        否则认为是数值属性
    """
    X = np.array(X)
    if X.ndim != 2:
        raise ValueError("X 必须是二维数组，当前形状为: {}".format(X.shape))

    n_samples, n_features = X.shape
    numeric_count = 0
    categorical_count = 0

    for j in range(n_features):
        col = X[:, j]

        if np.issubdtype(col.dtype, np.number):
            # 去掉 NaN
            if np.issubdtype(col.dtype, np.floating):
                col_no_nan = col[~np.isnan(col)]
            else:
                col_no_nan = col
            uniq = np.unique(col_no_nan).size if col_no_nan.size > 0 else 0
            cat_threshold = min(max_cat_unique, max(1, int(cat_ratio * n_samples)))

            if uniq <= cat_threshold:
                categorical_count += 1
            else:
                numeric_count += 1
        else:
            categorical_count += 1

    return numeric_count, categorical_count


def analyze_labels(y):
    """
    分析异常数量和比例：
    - 标签包含 {0,1} 时，默认 1 为异常
    - 否则，把样本数量最少的那一类视为异常
    """
    y = np.array(y).reshape(-1)
    n = y.shape[0]

    unique, counts = np.unique(y, return_counts=True)

    if unique.size == 1:
        # 只有一个类别，无法区分异常
        return 0, 0.0, unique[0]

    # 优先识别 {0,1}
    if unique.size == 2 and set(unique.tolist()) == {0, 1}:
        anomaly_label = 1
        anomaly_count = counts[unique.tolist().index(anomaly_label)]
        ratio = anomaly_count / float(n)
        return anomaly_count, ratio, anomaly_label

    # 一般情况：样本数最少的类视为异常
    min_idx = counts.argmin()
    anomaly_label = unique[min_idx]
    anomaly_count = counts[min_idx]
    ratio = anomaly_count / float(n)
    return anomaly_count, ratio, anomaly_label


def analyze_dataset(file_path: str):
    """给定一个绝对路径，打印数据集信息。"""
    if not os.path.isfile(file_path):
        print("错误：文件不存在 ->", file_path)
        return

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1].lower()

    # 读数据
    if ext == ".npz":
        X, y = load_npz(file_path)
    elif ext == ".mat":
        X, y = load_mat_generic(file_path)
    else:
        print("错误：暂不支持的文件扩展名：", ext)
        return

    X = np.array(X)
    y = np.array(y).reshape(-1)

    if X.shape[0] != y.shape[0]:
        print("错误：X 和 y 的样本数不一致：X.shape = {}, y.shape = {}".format(X.shape, y.shape))
        return

    n_samples, n_features = X.shape
    numeric_count, categorical_count = infer_feature_types(X)
    anomaly_count, anomaly_ratio, anomaly_label = analyze_labels(y)

    print("======================================")
    print("数据集名称       :", dataset_name)
    print("文件路径         :", file_path)
    print("--------------------------------------")
    print("样本总数 (n)     :", n_samples)
    print("属性总数量 (d)   :", n_features)
    print("数值属性数量     :", numeric_count)
    print("类别属性数量     :", categorical_count)
    print("--------------------------------------")
    print("标签取值         :", np.unique(y))
    print("被视为异常的标签 :", anomaly_label)
    print("异常值数量       :", anomaly_count)
    print("异常值比例       : {:.4f}".format(anomaly_ratio))
    print("======================================")


def main():
    # ================== 在这里改数据集路径 ==================
    # 示例 1：ADBench 自带 npz
    # file_path = r"G:\PaperCode\ADBench-main\adbench\datasets\Classical\6_cardio.npz"

    # 示例 2：你的 arrhythmia.mat
    # file_path = r"G:\OutlierDecetion\QMGOD\examples\dataset\arrhythmia.mat"

    file_path = r"G:\\PaperCode\\ADBench-main\\adbench\\datasets\\CV_by_ResNet18\\MVTec-AD_hazelnut.npz"  # 路径
    # =====================================================

    analyze_dataset(file_path)


if __name__ == "__main__":
    main()
