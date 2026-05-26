"""
cluster.py
功能：读取 cleaned_coins.json → 肘部法选 K → KMeans 聚类 → PCA 降维 → 保存结果
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # 无界面服务器环境，不弹窗，直接存图片
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# 1. 设置中文字体为文泉驿微米黑
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'sans-serif']

# 2. 解决保存图像时负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# ── 配置区 ────────────────────────────────────────────────────────────────────
INPUT_FILE   = "../data/cleaned_coins.json"
OUTPUT_FILE  = "../data/clustered_coins.json"
ELBOW_IMAGE  = "../data/elbow_curve.png"    # 肘部曲线图保存路径

# 标准化后的特征列（供 KMeans 和 PCA 使用）
SCALED_COLS = [
    "current_price_scaled",
    "market_cap_scaled",
    "total_volume_scaled",
    "price_change_percentage_24h_scaled",
    "market_cap_rank_scaled",
]

K_MIN = 2          # 肘部法扫描起点
K_MAX = 10         # 肘部法扫描终点
RANDOM_STATE = 42  # 固定随机种子，保证每次结果一致
# ──────────────────────────────────────────────────────────────────────────────


def load_cleaned(filepath: str) -> pd.DataFrame:
    """读取 cleaned_coins.json → DataFrame"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filepath)

    print(f"📂 读取文件：{abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    print(f"   数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
    return df


# ── 第一步：肘部法 ────────────────────────────────────────────────────────────

def elbow_method(X: np.ndarray) -> int:
    """
    对 K=2~10 依次跑 KMeans，记录 inertia（簇内误差平方和）。
    inertia 越小越好，但 K 越大 inertia 必然越小（过拟合）。
    拐点（下降速度突然变慢的地方）就是最佳 K。

    自动用"二阶差分最大值"找拐点，同时把曲线图存为 PNG。
    返回推荐的 K 值。
    """
    print(f"\n📐 肘部法：扫描 K = {K_MIN} ~ {K_MAX}...")
    inertias = []

    for k in range(K_MIN, K_MAX + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"   K={k:2d}  inertia={km.inertia_:.2f}")

    # ── 自动找拐点：二阶差分最大处 ──────────────────────────────────────────
    # 一阶差分：相邻 inertia 的下降量
    deltas = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    # 二阶差分：下降量本身的变化（变化最大 = 下降速度突然减慢 = 拐点）
    second_deltas = [deltas[i] - deltas[i + 1] for i in range(len(deltas) - 1)]
    best_k = K_MIN + 1 + int(np.argmax(second_deltas))
    print(f"\n   📌 自动推荐 K = {best_k}（二阶差分最大处）")

    # ── 画图 ──────────────────────────────────────────────────────────────────
    _plot_elbow(list(range(K_MIN, K_MAX + 1)), inertias, best_k)
    return best_k


def _plot_elbow(ks: list, inertias: list, best_k: int) -> None:
    """画肘部曲线并保存为 PNG"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, inertias, marker="o", color="#4C72B0", linewidth=2, label="Inertia")
    ax.axvline(x=best_k, color="#DD4444", linestyle="--", linewidth=1.5,
               label=f"推荐 K = {best_k}")
    ax.scatter([best_k], [inertias[best_k - K_MIN]],
               color="#DD4444", s=100, zorder=5)

    ax.set_title("KMeans 肘部曲线（Elbow Method）", fontsize=14)
    ax.set_xlabel("K（簇数）")
    ax.set_ylabel("Inertia（簇内误差平方和）")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, ELBOW_IMAGE)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   📊 肘部曲线图已保存：{save_path}")


# ── 第二步：KMeans 聚类 ───────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """
    用确定好的 K 跑 KMeans，返回每行的 cluster 标签（0, 1, 2...）
    n_init=10：随机初始化 10 次，取最优结果，避免陷入局部最优。
    """
    print(f"\n🔵 KMeans 聚类（K={k}，n_init=10）...")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)

    # 打印每个 cluster 的样本数
    unique, counts = np.unique(labels, return_counts=True)
    print("   各 cluster 样本数：")
    for cluster_id, count in zip(unique, counts):
        bar = "█" * (count // 10)
        print(f"   cluster {cluster_id}: {count:4d} 个  {bar}")

    return labels


# ── 第三步：PCA 降维 ──────────────────────────────────────────────────────────

def run_pca(X: np.ndarray) -> tuple[np.ndarray, PCA]:
    """
    PCA 把 5 维特征压缩成 2 维（x, y），供前端散点图使用。

    为什么 2 维？
    - 5 个特征人眼无法直接"看见"
    - PCA 找出方差最大的两个方向投影，尽量保留原始信息
    - 损失一部分精度，换来可视化能力

    返回 (降维后的坐标数组, pca对象)
    """
    print("\n🔻 PCA 降维（5维 → 2维）...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)   # shape: (n_coins, 2)

    explained = pca.explained_variance_ratio_
    print(f"   PC1 解释方差：{explained[0]:.1%}")
    print(f"   PC2 解释方差：{explained[1]:.1%}")
    print(f"   合计保留信息：{sum(explained):.1%}")

    return coords, pca


# ── 第四步：合并结果 & 保存 ───────────────────────────────────────────────────

def merge_and_save(df: pd.DataFrame, labels: np.ndarray,
                   coords: np.ndarray, filepath: str) -> None:
    """
    把 cluster 标签、PCA 坐标写回 DataFrame，再存为 JSON。
    只保留前端需要的列，不输出 _scaled 列（减小文件体积）。
    """
    df = df.copy()
    df["cluster"] = labels.tolist()
    df["pca_x"]   = np.round(coords[:, 0], 6).tolist()
    df["pca_y"]   = np.round(coords[:, 1], 6).tolist()

    # 只输出前端需要的列
    original_feature_cols = [
        "current_price",
        "market_cap",
        "total_volume",
        "price_change_percentage_24h",
        "market_cap_rank",
    ]
    meta_cols = ["id", "symbol", "name", "image"]
    result_cols = meta_cols + original_feature_cols + ["cluster", "pca_x", "pca_y"]

    # 只保留存在的列（image 可能没有）
    result_cols = [c for c in result_cols if c in df.columns]
    df_out = df[result_cols]

    # 保存
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    records = df_out.to_dict(orient="records")
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n💾 聚类结果已保存：{abs_path}")
    print(f"   共 {len(records)} 条，字段：{result_cols}")


def print_cluster_summary(df: pd.DataFrame, labels: np.ndarray) -> None:
    """打印每个 cluster 的特征均值，帮助理解各群组特征"""
    df = df.copy()
    df["cluster"] = labels

    feature_cols = [
        "current_price",
        "market_cap",
        "total_volume",
        "price_change_percentage_24h",
        "market_cap_rank",
    ]
    existing = [c for c in feature_cols if c in df.columns]

    print("\n📋 各 cluster 特征均值：")
    summary = df.groupby("cluster")[existing].mean().round(2)
    print(summary.to_string())


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  cluster.py  KMeans 聚类 + PCA 降维")
    print("=" * 55)

    # ① 读取数据
    df = load_cleaned(INPUT_FILE)

    # ② 提取标准化特征矩阵
    missing = [c for c in SCALED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"找不到标准化列，请先运行 clean_data.py：{missing}")
    X = df[SCALED_COLS].values   # numpy 数组，shape: (n_coins, 5)

    # ③ 肘部法确定最佳 K
    best_k = elbow_method(X)

    # ④ 允许手动覆盖 K（看完肘部图觉得推荐值不对时，改这里）
    # best_k = 5   # ← 取消注释并改成你认为合适的值

    # ⑤ KMeans 聚类
    labels = run_kmeans(X, best_k)

    # ⑥ PCA 降维
    coords, _ = run_pca(X)

    # ⑦ 打印各群组均值
    print_cluster_summary(df, labels)

    # ⑧ 合并保存
    merge_and_save(df, labels, coords, OUTPUT_FILE)

    print("\n✅ cluster.py 执行完毕，下一步运行 export_json.py")
    print("=" * 55)