"""IO helpers: JSON + compressed NumPy arrays, with directory creation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def save_arrays(path: str | Path, **arrays: np.ndarray) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(p, **arrays)


def load_arrays(path: str | Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
