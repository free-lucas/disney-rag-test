from __future__ import annotations

from pathlib import Path

import numpy as np


def write_index(index: object, path: str | Path) -> Path:
    import faiss

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob = faiss.serialize_index(index)
    destination.write_bytes(np.asarray(blob, dtype='uint8').tobytes())
    return destination


def read_index(path: str | Path) -> object:
    import faiss

    blob = np.frombuffer(Path(path).read_bytes(), dtype='uint8').copy()
    return faiss.deserialize_index(blob)
