"""run.py ── 一键自动化流水线
用法：在项目根目录执行 python run.py
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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
DATA_DIR = os.path.join(ROOT_DIR, "data")
SERVER_PORT = 8080

PIPELINE = [
    ("① 数据获取", "fetch_data.py"),
    ("② 数据清洗", "clean_data.py"),
    ("③ 聚类+降维", "cluster.py"),
    ("④ 输出JSON", "export_json.py"),
]

# ──────────────────────────────────────────────────────────────
def print_banner():
    print("\n" + "═" * 60)
    print(" 🚀 Crypto Cluster 一键自动化流水线")
    print("═" * 60 + "\n")

def run_step(label: str, script: str) -> bool:
    script_path = os.path.join(SCRIPTS_DIR, script)
    print(f"\n{'─'*60}")
    print(f" {label} → {script}")
    print(f"{'─'*60}")
    start = time.time()
    result = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"\n ✅ 完成 ({elapsed:.1f}s)")
        return True
    else:
        print(f"\n ❌ 失败（退出码 {result.returncode}）")
        print(f" 请修复 {script} 后重新运行。")
        return False

def open_browser_after_delay(port: int, delay: float = 1.5):
    """延迟打开浏览器，等服务器启动"""
    time.sleep(delay)
    # 打开 frontend/index.html（根目录启动服务器）
    url = f"http://localhost:{port}/frontend/index.html"
    print(f"\n 🌐 浏览器打开：{url}")
    webbrowser.open(url)

def start_server():
    """在项目根目录启动 Python 静态服务器"""
    print(f"\n{'═'*60}")
    print(f" 🖥️ 启动本地服务器（端口 {SERVER_PORT}）")
    print(f" 按 Ctrl+C 停止服务器")
    print(f"{'═'*60}\n")

    # 在后台线程中打开浏览器
    t = threading.Thread(target=open_browser_after_delay, args=(SERVER_PORT,), daemon=True)
    t.start()

    # 切换到项目根目录（确保能同时访问 frontend/ 和 data/）
    os.chdir(ROOT_DIR)

    try:
        subprocess.run([sys.executable, "-m", "http.server", str(SERVER_PORT)])
    except KeyboardInterrupt:
        print("\n\n 服务器已停止。")

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
        print(f" ❌ 流水线在 {failed_at} 处中断（{total_elapsed:.1f}s）")
        print("═" * 60 + "\n")
        sys.exit(1)
    print(f"\n{'═'*60}")
    print(f" 🎉 全部步骤完成！总耗时：{total_elapsed:.1f}s")
    print("═" * 60)
    if not os.path.isdir(FRONTEND_DIR):
        print(f"\n ⚠️ 找不到 frontend/ 目录")
        sys.exit(0)
    if not os.path.isfile(os.path.join(DATA_DIR, "result.json")):
        print(f"\n ⚠️ 找不到 data/result.json，请检查 export_json.py")
        sys.exit(0)
    start_server()

if __name__ == "__main__":
    main()