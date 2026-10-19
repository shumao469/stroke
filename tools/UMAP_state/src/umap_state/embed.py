from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

@dataclass
class UMAPConfig:
    n_components: int = 2
    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 0
    use_gpu: bool = False

def _get_umap_backend(use_gpu: bool):
    if use_gpu:
        # RAPIDS cuML UMAP if available
        try:
            from cuml.manifold import UMAP as cuUMAP
            return "cuml", cuUMAP
        except Exception:
            pass
    # CPU umap-learn
    from umap import UMAP as cpuUMAP
    return "umap-learn", cpuUMAP

def fit_umap(X: np.ndarray, cfg: UMAPConfig) -> Any:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    backend_name, UMAPCls = _get_umap_backend(cfg.use_gpu)
    model = UMAPCls(
        n_components=int(cfg.n_components),
        n_neighbors=int(cfg.n_neighbors),
        min_dist=float(cfg.min_dist),
        metric=str(cfg.metric),
        random_state=int(cfg.random_state),
    )
    Z = model.fit_transform(X)
    return model, np.asarray(Z, dtype=np.float32), backend_name

def transform_umap(model: Any, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not hasattr(model, "transform"):
        raise RuntimeError("UMAP backend does not support transform(). Re-fit on combined data instead.")
    return np.asarray(model.transform(X), dtype=np.float32)
