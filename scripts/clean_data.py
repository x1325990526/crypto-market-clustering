"""
clean_data.py
功能：读取 raw_coins.json → 清洗缺失值 → 标准化特征 → 保存 cleaned_coins.json
"""

import json
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 配置区 ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "../data/raw_coins.json"    # 上一步的输出
OUTPUT_FILE = "../data/cleaned_coins.json"

# 聚类要用到的 5 个特征列（量纲差异很大，所以必须标准化）
FEATURE_COLS = [
    "current_price",                  # 当前价格（美元）
    "market_cap",                     # 市值（数十亿级别）
    "total_volume",                   # 24h 交易量
    "price_change_percentage_24h",    # 24h 涨跌幅（-100 ~ +数百）
    "market_cap_rank",                # 市值排名（1 ~ 1000）
]

# 保留的元信息列（不参与聚类，但要带入后续步骤）
META_COLS = ["id", "symbol", "name", "image"]
# ──────────────────────────────────────────────────────────────────────────────


def load_raw(filepath: str) -> pd.DataFrame:
    """
    读取 raw_coins.json，提取 coins 列表，转成 DataFrame。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filepath)

    print(f"📂 读取文件：{abs_path}")
    with open(abs_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # raw_coins.json 的结构是 { "coins": [...], "total": ..., ... }
    coins = raw.get("coins", [])
    df = pd.DataFrame(coins)
    print(f"   原始数据：{len(df)} 行 × {len(df.columns)} 列")
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    只保留需要的列，多余字段全部丢掉，减少后续干扰。
    """
    keep = META_COLS + FEATURE_COLS

    # 检查是否有列名对不上
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"以下列在数据中不存在，请检查字段名：{missing}")

    df = df[keep].copy()
    print(f"   保留列：{keep}")
    return df


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除特征列中含有空值的行。
    元信息列（如 image）有空也无所谓，只关心特征列。
    """
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"   🗑️  删除含空值的行：{dropped} 行（剩余 {after} 行）")
    else:
        print(f"   ✅ 无空值行，全部保留（{after} 行）")
    return df.reset_index(drop=True)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 id 去重，防止同一币种抓取多次。
    """
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    after = len(df)
    if before != after:
        print(f"   🗑️  删除重复行：{before - after} 行（剩余 {after} 行）")
    else:
        print(f"   ✅ 无重复行")
    return df.reset_index(drop=True)


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 5 个特征列做 Z-score 标准化（均值=0，标准差=1）。

    为什么要标准化？
    - market_cap 单位是万亿，price_change_percentage_24h 只是个位数
    - 不标准化的话，KMeans 会被市值"统治"，其他特征几乎没有影响力
    - StandardScaler 让每个特征的重要性对等

    标准化后的列名加 "_scaled" 后缀，原始数值列保留（后续输出 JSON 需要）。
    """
    scaler = StandardScaler()

    # fit_transform 返回 numpy 数组，转回 DataFrame
    scaled_values = scaler.fit_transform(df[FEATURE_COLS])
    scaled_cols = [f"{col}_scaled" for col in FEATURE_COLS]
    scaled_df = pd.DataFrame(scaled_values, columns=scaled_cols, index=df.index)

    # 拼回原 DataFrame（原始列 + 标准化列并存）
    df = pd.concat([df, scaled_df], axis=1)

    print(f"\n📊 标准化完成，新增列：{scaled_cols}")
    print("\n   各特征标准化后的统计（应该接近 均值≈0，标准差≈1）：")
    print(df[scaled_cols].describe().round(4).to_string())
    return df


def print_sample(df: pd.DataFrame, n: int = 3) -> None:
    """
    打印前 n 行的关键信息，快速检查结果是否正常。
    """
    print(f"\n🔍 前 {n} 条数据预览：")
    preview_cols = ["name", "market_cap_rank"] + FEATURE_COLS[:3]
    print(df[preview_cols].head(n).to_string(index=False))


def save_cleaned(df: pd.DataFrame, filepath: str) -> None:
    """
    把清洗后的 DataFrame 存为 JSON。
    orient="records" 让每行变成一个 dict，格式和 raw_coins.json 一致。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filepath)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    # DataFrame → list of dict → JSON
    records = df.to_dict(orient="records")
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n💾 清洗结果已保存：{abs_path}")
    print(f"   共 {len(records)} 条，包含原始特征列 + _scaled 标准化列")


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  clean_data.py  数据清洗 & 标准化")
    print("=" * 55)

    # ① 读取原始数据
    df = load_raw(INPUT_FILE)

    # ② 只保留需要的列
    df = select_columns(df)

    # ③ 删除特征列含空值的行
    df = drop_nulls(df)

    # ④ 去除重复币种
    df = drop_duplicates(df)

    # ⑤ 标准化
    df = standardize(df)

    # ⑥ 预览
    print_sample(df)

    # ⑦ 保存
    save_cleaned(df, OUTPUT_FILE)

    print("\n✅ clean_data.py 执行完毕，下一步运行 cluster.py")
    print("=" * 55)