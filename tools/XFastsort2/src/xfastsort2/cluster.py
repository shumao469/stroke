from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans

@dataclass
class ClusterConfig:
    n_clusters: int = 3
    random_state: int = 0

def cluster_kmeans(X: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """KMeans clustering on embedded features."""
    if X.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    km = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_state, n_init="auto")
    return km.fit_predict(X).astype(np.int64)
