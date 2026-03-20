"""兼容入口：转发到 experiments/analysis/simple_feature_engineering.py。"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).resolve().parent / "experiments/analysis/simple_feature_engineering.py", run_name="__main__")
