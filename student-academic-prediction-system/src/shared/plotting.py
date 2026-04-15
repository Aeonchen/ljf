from pathlib import Path
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib import font_manager
import matplotlib.pyplot as plt


_FALLBACK_FONT_NAMES = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "Source Han Sans SC",
    "PingFang SC",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def _load_chinese_font():
    candidate_font_paths = [
        # Windows
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        # Linux 常见字体
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        # macOS 常见字体
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
    ]

    for font_path in candidate_font_paths:
        if not os.path.exists(font_path):
            continue

        font_manager.fontManager.addfont(font_path)
        font_prop = font_manager.FontProperties(fname=font_path)
        font_name = font_prop.get_name()

        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [font_name]
        matplotlib.rcParams["axes.unicode_minus"] = False

        print(f"✅ 已加载中文字体: {font_name}")
        return font_prop

    # 如果未命中具体文件路径，回退为常见中文字体名，让 Matplotlib 自行匹配。
    fallback_names = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = fallback_names
    matplotlib.rcParams["axes.unicode_minus"] = False
    print("⚠️ 未找到指定中文字体文件，已启用字体名回退列表")
    return font_manager.FontProperties(family=fallback_names)


ZH_FONT = _load_chinese_font()


def apply_plot_defaults():
    """统一设置绘图字体，避免被 seaborn/style 覆盖成 Arial。"""
    primary_font = ZH_FONT.get_name() if ZH_FONT is not None else _FALLBACK_FONT_NAMES[0]
    merged = [primary_font, *_FALLBACK_FONT_NAMES]
    # 去重并保持顺序
    merged = list(dict.fromkeys(merged))

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = merged
    matplotlib.rcParams["axes.unicode_minus"] = False


apply_plot_defaults()


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
