"""路径与目录管理工具。"""

from pathlib import Path

DEFAULT_DIRS = (
    'data',
    'models',
    'reports',
    'notebooks',
    'src',
    'experiments',
    'logs',
    'configs',
    'tests',
    'src/regression',
    'src/warning',
    'src/shared',
    'experiments/regression',
    'experiments/warning',
    'experiments/analysis',
)


def ensure_directories(base_path='.', directories=None):
    base = Path(base_path)
    created = []
    for directory in directories or DEFAULT_DIRS:
        path = base / directory
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    return created
