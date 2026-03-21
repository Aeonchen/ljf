"""兼容入口：转发到 experiments/analysis/check_features.py。"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).resolve().parent / "experiments/analysis/check_features.py", run_name="__main__")
