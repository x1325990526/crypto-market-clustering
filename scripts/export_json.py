"""
export_json.py
功能：读取 clustered_coins.json → 计算每个 cluster 的统计摘要 → 输出 result.json（前端唯一数据源）
"""

import json
import os
import numpy as np
import pandas as pd

# ── 配置区 ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "../data/clustered_coins.json"
OUTPUT_FILE = "../data/result.json"

# 前端雷达图 / 表格要展示的原始特征
FEATURE_COLS = [
    "current_price",
    "market_cap",
    "total_volume",
    "price_change_percentage_24h",
    "market_cap_rank",
]

# 雷达图各维度的中文标签（顺序与 FEATURE_COLS 一致）
FEATURE_LABELS = ["价格(USD)", "市值", "交易量(24h)", "涨跌幅(24h)%", "市值排名"]

# cluster 群组描述（根据特征均值人工填写，先留占位，export 后再改）
CLUSTER_DESC = {
    0: "待分析群组 0",
    1: "待分析群组 1",
    2: "待分析群组 2",
    3: "待分析群组 3",
    4: "待分析群组 4",
}
# ──────────────────────────────────────────────────────────────────────────────


def load_clustered(filepath: str) -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filepath)
    print(f"📂 读取文件：{abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    print(f"   数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
    return df


def build_coins_list(df: pd.DataFrame) -> list:
    """
    构建前端散点图 / 表格用的 coins 数组。
    只保留前端渲染需要的字段，精简体积。
    """
    cols = ["id", "symbol", "name", "image",
            "cluster", "pca_x", "pca_y"] + FEATURE_COLS
    cols = [c for c in cols if c in df.columns]

    coins = []
    for _, row in df[cols].iterrows():
        coin = {}
        for c in cols:
            val = row[c]
            # NaN → None（JSON 序列化安全）
            coin[c] = None if (isinstance(val, float) and np.isnan(val)) else val
        coins.append(coin)
    return coins


def build_cluster_stats(df: pd.DataFrame) -> list:
    """
    计算每个 cluster 的统计数据，供雷达图使用。
    返回格式：
    [
      {
        "cluster": 0,
        "count": 200,
        "description": "...",
        "features": {
          "current_price":  { "mean": ..., "min": ..., "max": ..., "label": "价格(USD)" },
          ...
        },
        "top_coins": ["bitcoin", "ethereum", ...]   ← 市值排名最靠前的5个
      },
      ...
    ]
    """
    stats = []
    for cluster_id, group in df.groupby("cluster"):
        feature_stats = {}
        for col, label in zip(FEATURE_COLS, FEATURE_LABELS):
            if col not in group.columns:
                continue
            vals = group[col].dropna()
            feature_stats[col] = {
                "label": label,
                "mean":  round(float(vals.mean()), 4),
                "median": round(float(vals.median()), 4),
                "min":   round(float(vals.min()), 4),
                "max":   round(float(vals.max()), 4),
            }

        # 市值排名最靠前的 5 个币（rank 越小越靠前）
        top5 = (group.nsmallest(5, "market_cap_rank")["name"].tolist()
                if "market_cap_rank" in group.columns else [])

        stats.append({
            "cluster":     int(cluster_id),
            "count":       int(len(group)),
            "description": CLUSTER_DESC.get(int(cluster_id), f"群组 {cluster_id}"),
            "features":    feature_stats,
            "top_coins":   top5,
        })

    # 按 cluster id 排序
    stats.sort(key=lambda x: x["cluster"])
    return stats


def print_summary(cluster_stats: list) -> None:
    """在终端打印一份人类可读的摘要，帮助填写 CLUSTER_DESC。"""
    print("\n" + "=" * 65)
    print("  📋  各 cluster 特征摘要（用于填写 CLUSTER_DESC）")
    print("=" * 65)
    for s in cluster_stats:
        print(f"\nCluster {s['cluster']}  ({s['count']} 个币)")
        print(f"  代表币种：{', '.join(s['top_coins'])}")
        for col, info in s["features"].items():
            print(f"  {info['label']:20s}  均值={info['mean']:.4g}"
                  f"  中位数={info['median']:.4g}"
                  f"  范围=[{info['min']:.4g}, {info['max']:.4g}]")


def save_result(coins: list, cluster_stats: list, filepath: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    output = {
        "meta": {
            "total_coins":    len(coins),
            "total_clusters": len(cluster_stats),
            "feature_cols":   FEATURE_COLS,
            "feature_labels": FEATURE_LABELS,
        },
        "cluster_stats": cluster_stats,
        "coins":          coins,
    }

    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(abs_path) / 1024
    print(f"\n💾 前端数据已保存：{abs_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    print("=" * 55)
    print("  export_json.py  聚合统计 → result.json")
    print("=" * 55)

    df = load_clustered(INPUT_FILE)

    coins = build_coins_list(df)
    cluster_stats = build_cluster_stats(df)

    print_summary(cluster_stats)
    save_result(coins, cluster_stats, OUTPUT_FILE)

    print("\n✅ export_json.py 执行完毕，下一步打开 frontend/index.html")
    print("=" * 55)