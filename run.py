"""
run.py  ── 一键自动化流水线
用法：在项目根目录执行  python run.py
会依次运行：fetch_data → clean_data → cluster → export_json
完成后自动启动本地服务器，浏览器打开前端页面。
"""

import subprocess
import sys
import os
import time
import webbrowser
import threading

# ── 配置 ──────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))   # 项目根目录
SCRIPTS_DIR  = os.path.join(ROOT_DIR, "scripts")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
SERVER_PORT  = 8080

PIPELINE = [
    ("① 数据获取",  "fetch_data.py"),
    ("② 数据清洗",  "clean_data.py"),
    ("③ 聚类+降维", "cluster.py"),
    ("④ 输出JSON",  "export_json.py"),
]
# ──────────────────────────────────────────────────────────────


def print_banner():
    print("\n" + "═" * 60)
    print("   🚀  Crypto Cluster  一键自动化流水线")
    print("═" * 60 + "\n")


def run_step(label: str, script: str) -> bool:
    script_path = os.path.join(SCRIPTS_DIR, script)
    print(f"\n{'─'*60}")
    print(f"  {label}  →  {script}")
    print(f"{'─'*60}")

    start = time.time()
    result = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✅ 完成  ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n  ❌ 失败（退出码 {result.returncode}）")
        print(f"     请修复 {script} 后重新运行。")
        return False


def open_browser_after_delay(url: str, delay: float = 1.5):
    time.sleep(delay)
    print(f"\n  🌐 浏览器打开：{url}")
    webbrowser.open(url)


def start_server():
    # 服务器根目录 = 项目根目录
    # index.html 在 frontend/index.html
    # result.json 在 data/result.json
    # index.html 里 fetch('../data/result.json') → /data/result.json ✅
    url = f"http://localhost:{SERVER_PORT}/frontend/index.html"

    print(f"\n{'═'*60}")
    print(f"  🖥️  启动本地服务器（端口 {SERVER_PORT}）")
    print(f"  服务根目录：{ROOT_DIR}")
    print(f"  访问地址：  {url}")
    print(f"  按 Ctrl+C 停止服务器")
    print(f"{'═'*60}\n")

    # 后台线程打开浏览器
    t = threading.Thread(target=open_browser_after_delay, args=(url,), daemon=True)
    t.start()

    # 从项目根目录启动服务器（关键修复）
    os.chdir(ROOT_DIR)
    try:
        subprocess.run([sys.executable, "-m", "http.server", str(SERVER_PORT)])
    except KeyboardInterrupt:
        print("\n\n  服务器已停止。")


def main():
    print_banner()

    total_start = time.time()
    failed_at = None

    for label, script in PIPELINE:
        success = run_step(label, script)
        if not success:
            failed_at = script
            break

    total_elapsed = time.time() - total_start

    if failed_at:
        print(f"\n{'═'*60}")
        print(f"  ❌ 流水线在 {failed_at} 处中断（{total_elapsed:.1f}s）")
        print("═" * 60 + "\n")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print(f"  🎉 全部步骤完成！总耗时：{total_elapsed:.1f}s")
    print("═" * 60)

    if not os.path.isdir(FRONTEND_DIR):
        print(f"\n  ⚠️  找不到 frontend/ 目录")
        sys.exit(0)

    start_server()


if __name__ == "__main__":
    main()