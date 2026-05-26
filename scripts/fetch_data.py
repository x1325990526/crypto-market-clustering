"""
fetch_data.py
功能：调用 CoinGecko /coins/markets 接口，分页抓取币种数据，存为 JSON 文件。
"""

import requests
import time
import json
import os
from datetime import datetime

# ── 配置区（需要修改的参数都在这里）────────────────────────────────────────
BASE_URL   = "https://api.coingecko.com/api/v3/coins/markets"
VS_CURRENCY = "usd"          # 计价货币
PER_PAGE   = 250             # 每页最多 250 条（CoinGecko 上限）
TOTAL_PAGES = 3              # 抓取页数：4 页 × 250 = 1000 条
SLEEP_SEC  = 1.5             # 每次请求之间的等待秒数（避免被限速）
OUTPUT_DIR = "../data"       # 输出目录（相对于 scripts/ 的路径）
OUTPUT_FILE = "raw_coins.json"

# 抓取的字段（CoinGecko 会自动返回，这里只是说明用）
# id, symbol, name, current_price, market_cap, total_volume,
# price_change_percentage_24h, market_cap_rank
# ──────────────────────────────────────────────────────────────────────────────


def fetch_page(page: int) -> list:
    """
    抓取单页数据，返回列表。
    失败时打印错误并返回空列表。
    """
    params = {
        "vs_currency": VS_CURRENCY,
        "order":       "market_cap_desc",   # 按市值从大到小排序
        "per_page":    PER_PAGE,
        "page":        page,
        "sparkline":   "false",             # 不要折线图数据，减小体积
        "price_change_percentage": "24h",   # 附带 24h 涨跌幅
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)

        # 检查 HTTP 状态码
        if response.status_code == 429:
            print(f"  ⚠️  第 {page} 页：请求过频（429），等待 10 秒后重试...")
            time.sleep(10)
            return fetch_page(page)   # 递归重试一次

        response.raise_for_status()   # 其他 4xx/5xx 直接抛出异常
        data = response.json()
        print(f"  ✅ 第 {page} 页：成功获取 {len(data)} 条")
        return data

    except requests.exceptions.Timeout:
        print(f"  ❌ 第 {page} 页：请求超时，跳过")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  ❌ 第 {page} 页：请求失败 → {e}")
        return []


def fetch_all() -> list:
    """
    循环抓取所有页，合并成一个列表。
    """
    all_coins = []

    print(f"\n🚀 开始抓取数据（共 {TOTAL_PAGES} 页，每页 {PER_PAGE} 条）")
    print(f"   计价货币：{VS_CURRENCY.upper()}，每页间隔：{SLEEP_SEC}s\n")

    for page in range(1, TOTAL_PAGES + 1):
        print(f"📄 正在抓取第 {page}/{TOTAL_PAGES} 页...")
        coins = fetch_page(page)
        all_coins.extend(coins)

        # 最后一页不用等待
        if page < TOTAL_PAGES:
            time.sleep(SLEEP_SEC)

    print(f"\n📦 全部抓取完成，共 {len(all_coins)} 条原始数据")
    return all_coins


def save_json(data: list) -> str:
    """
    把数据保存为 JSON 文件，同时加上抓取时间戳。
    返回保存的文件路径。
    """
    # 确保输出目录存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 组装带时间戳的完整数据结构
    output = {
        "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total":      len(data),
        "currency":   VS_CURRENCY,
        "coins":      data,
    }

    filepath = os.path.join(output_dir, OUTPUT_FILE)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"💾 数据已保存到：{filepath}")
    return filepath


def preview(data: list, n: int = 3) -> None:
    """
    打印前 n 条数据的关键字段，用于快速校验。
    """
    print(f"\n🔍 预览前 {n} 条数据：")
    keys = ["id", "name", "current_price", "market_cap",
            "total_volume", "price_change_percentage_24h", "market_cap_rank"]

    for coin in data[:n]:
        row = {k: coin.get(k) for k in keys}
        print(f"   {row}")


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    coins = fetch_all()

    if not coins:
        print("\n⚠️  没有抓取到任何数据，请检查网络或 API 状态。")
    else:
        preview(coins)
        save_json(coins)
        print("\n✅ fetch_data.py 执行完毕，下一步运行 clean_data.py")