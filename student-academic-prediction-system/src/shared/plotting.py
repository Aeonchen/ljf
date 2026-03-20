"""统一绘图保存与关闭行为。"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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
