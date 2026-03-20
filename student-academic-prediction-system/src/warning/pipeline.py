"""学业预警主流程兼容层。"""

from run_warning_optimized import main as legacy_warning_main



def main():
    """统一的预警训练入口。"""
    legacy_warning_main()
