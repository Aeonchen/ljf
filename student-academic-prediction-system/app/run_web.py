"""
启动Web应用的便捷脚本
"""
import subprocess
import sys


def main():
    """启动Streamlit应用"""
    print("🚀 启动学生学业预警系统Web应用...")
    print("正在启动Streamlit服务器...")
    print("请在浏览器中打开: http://localhost:8501")
    print("按 Ctrl+C 停止服务器")

    # 运行Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "web_app.py"])


if __name__ == "__main__":
    main()