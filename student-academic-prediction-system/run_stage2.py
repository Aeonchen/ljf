"""兼容入口：转发到 experiments/regression/run_stage2.py。"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(Path(__file__).resolve().parent / "experiments/regression/run_stage2.py", run_name="__main__")
