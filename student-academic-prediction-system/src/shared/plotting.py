from pathlib import Path
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib import font_manager
import matplotlib.pyplot as plt


def _setup_chinese_font():
    """优先使用本机真实存在的中文字体文件"""
    candidate_fonts = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
    ]

    for font_path in candidate_fonts:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams["font.family"] = font_name
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            print(f"✅ 已加载中文字体: {font_name} -> {font_path}")
            return

    print("⚠️ 未找到可用中文字体，图片中的中文可能显示为方框")


_setup_chinese_font()


def save_figure(fig, path, dpi=300, bbox_inches='tight'):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches=bbox_inches)
    return str(output)


def safe_close(fig=None):
    if fig is not None:
        plt.close(fig)
    else:
        plt.close('all')