"""通用读写工具。"""

import json
from pathlib import Path

import joblib
import pandas as pd
from pandas.errors import ParserError


DEFAULT_ENCODINGS = ('utf-8', 'gbk', 'latin1')


def load_csv(path, encodings=None, **kwargs):
    encodings = encodings or DEFAULT_ENCODINGS
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, ParserError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, **kwargs)


def save_dataframe(df, path, **kwargs):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, **kwargs)
    return str(output)


def save_json(data, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return str(output)


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as file:
        return json.load(file)


def save_pickle(obj, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, output)
    return str(output)


def load_pickle(path):
    return joblib.load(path)


def save_text(text, path):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    return str(output)
