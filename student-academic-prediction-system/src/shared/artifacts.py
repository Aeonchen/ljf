"""训练产物 manifest 工具。"""

import json
from datetime import datetime
from pathlib import Path



def build_manifest(task, model_path, report_path, **extra):
    manifest = {
        'task': task,
        'model_path': model_path,
        'report_path': report_path,
        'generated_at': datetime.now().isoformat(),
    }
    manifest.update(extra)
    return manifest



def save_manifest(manifest, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return str(target)



def load_manifest(path):
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    with manifest_path.open('r', encoding='utf-8') as file:
        return json.load(file)
