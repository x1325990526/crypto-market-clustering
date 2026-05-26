"""
clean_data.py  v2
功能：读取 raw_coins.json → 清洗 → 对数变换（修复极端偏态）→ 标准化 → 保存
修复：market_cap / total_volume / current_price 分布极右偏，
     头部资产（如 BTC）的数值与尾部资产存在数万倍的物理差距。
     StandardScaler 无法解决离群值问题，需先做 log1p 变换再标准化。
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 配置区 ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "../data/raw_coins.json"
OUTPUT_FILE = "../data/cleaned_coins.json"

# 需要先做 log1p 变换的列（量纲极大、分布极右偏）
LOG_COLS = [
    "current_price",
    "market_cap",
    "total_volume",
]

# 聚类用的 5 个特征列（变换后再标准化）
FEATURE_COLS = [
    "current_price",
    "market_cap",
    "total_volume",
    "price_change_percentage_24h",
    "market_cap_rank",
]

META_COLS = ["id", "symbol", "name", "image"]
# ──────────────────────────────────────────────────────────────────────────────


def load_raw(filepath: str) -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filepath)
    print(f"📂 读取文件：{abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    coins = raw.get("coins", [])
    df = pd.DataFrame(coins)
    print(f"   原始数据：{len(df)} 行 × {len(df.columns)} 列")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = META_COLS + FEATURE_COLS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"以下列在数据中不存在：{missing}")
    return df[keep].copy()


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    after = len(df)
    print(f"   🗑️  删除含空值行：{before - after} 行（剩余 {after} 行）")
    return df.reset_index(drop=True)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    after = len(df)
    if before != after:
        print(f"   🗑️  删除重复行：{before - after} 行（剩余 {after} 行）")
    return df.reset_index(drop=True)


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    对极右偏列做 log1p 变换（log(1+x)），压缩极端值。

    为什么用 log1p？
    - BTC 市值 ≈ 1.5 万亿，普通小币 ≈ 100 万，相差 150 万倍
    - log1p(1_500_000_000_000) ≈ 28.0
    - log1p(1_000_000)         ≈ 14.0
    - 差距从 150 万倍 → 2 倍，StandardScaler 才能正常工作

    注意：price_change_percentage_24h 可能为负，不做变换。
    """
    df = df.copy()
    for col in LOG_COLS:
        df[col] = df[col].abs()
        df[col] = np.log1p(df[col])
    print(f"\n🔁 对数变换（log1p）已应用：{LOG_COLS}")

    print("   变换后各列分布（最大值/最小值之比应 <100）：")
    for col in LOG_COLS:
        col_min = df[col].min()
        col_max = df[col].max()
        ratio = col_max / col_min if col_min > 0 else float("inf")
        print(f"   {col:35s}  min={col_min:.2f}  max={col_max:.2f}  ratio={ratio:.1f}x")

    return df


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score 标准化，log 变换之后再做效果才正常。"""
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[FEATURE_COLS])
    scaled_cols = [f"{col}_scaled" for col in FEATURE_COLS]
    scaled_df = pd.DataFrame(scaled_values, columns=scaled_cols, index=df.index)
    df = pd.concat([df, scaled_df], axis=1)

    print(f"\n📊 标准化完成，新增列：{scaled_cols}")
    print("\n   各特征标准化后统计（均值应≈0，标准差应≈1）：")
    print(df[scaled_cols].describe().round(3).to_string())
    return df


def save_cleaned(df: pd.DataFrame, filepath: str) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    records = df.to_dict(orient="records")
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\n💾 清洗结果已保存：{abs_path}（{len(records)} 条）")


if __name__ == "__main__":
    print("=" * 55)
    print("  clean_data.py v2  数据清洗 + 对数变换 + 标准化")
    print("=" * 55)

    df = load_raw(INPUT_FILE)
    df = select_columns(df)
    df = drop_nulls(df)
    df = drop_duplicates(df)
    df = log_transform(df)      # ← 新增：先变换再标准化
    df = standardize(df)
    save_cleaned(df, OUTPUT_FILE)

    print("\n✅ clean_data.py 执行完毕，下一步运行 cluster.py")
    print("=" * 55)