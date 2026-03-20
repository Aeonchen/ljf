"""兼容入口：转发到 experiments/warning/binary_classification.py。"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).resolve().parent / "experiments/warning/binary_classification.py", run_name="__main__")
