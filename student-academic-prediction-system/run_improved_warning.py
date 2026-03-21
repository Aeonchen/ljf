"""兼容入口：转发到 experiments/warning/run_improved_warning.py。"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).resolve().parent / "experiments/warning/run_improved_warning.py", run_name="__main__")
