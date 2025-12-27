from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np

@dataclass
class ClusterConfig:
    method: str = "hdbscan"   # 'hdbscan' or 'kmeans'
    min_cluster_size: int = 50
    n_clusters: int = 5       # for kmeans fallback
    random_state: int = 0

def cluster_states(Z: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """Cluster UMAP embedding into discrete 'states'.

    Returns
    -------
    labels: (T,) int64
      -1 may appear for HDBSCAN noise points.
    """
    Z = np.asarray(Z, dtype=float)
    if cfg.method.lower() == "hdbscan":
        try:
            import hdbscan
        except Exception:
            # fallback
            cfg = ClusterConfig(method="kmeans", n_clusters=cfg.n_clusters, random_state=cfg.random_state)
        else:
            cl = hdbscan.HDBSCAN(min_cluster_size=int(cfg.min_cluster_size))
            lab = cl.fit_predict(Z)
            return lab.astype(np.int64)

    # KMeans fallback
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=int(cfg.n_clusters), random_state=int(cfg.random_state), n_init="auto")
    return km.fit_predict(Z).astype(np.int64)

def transition_matrix(labels: np.ndarray, n_states: Optional[int] = None, include_noise: bool = False) -> np.ndarray:
    """Compute state-to-state transition counts (row-normalized)."""
    lab = np.asarray(labels, dtype=int).reshape(-1)
    if not include_noise:
        m = lab >= 0
        lab = lab[m]
    if lab.size < 2:
        return np.zeros((0,0), dtype=np.float32)
    if n_states is None:
        n_states = int(lab.max()) + 1
    M = np.zeros((n_states, n_states), dtype=np.float64)
    for a, b in zip(lab[:-1], lab[1:]):
        if a < 0 or b < 0:
            continue
        if a >= n_states or b >= n_states:
            continue
        M[a, b] += 1.0
    # row-normalize
    row = M.sum(axis=1, keepdims=True) + 1e-12
    P = (M / row).astype(np.float32)
    return P

def state_occupancy(labels: np.ndarray, n_states: Optional[int] = None) -> np.ndarray:
    lab = np.asarray(labels, dtype=int).reshape(-1)
    m = lab >= 0
    lab = lab[m]
    if lab.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if n_states is None:
        n_states = int(lab.max()) + 1
    occ = np.zeros((n_states,), dtype=np.float64)
    for s in range(n_states):
        occ[s] = float((lab == s).sum())
    occ = occ / (occ.sum() + 1e-12)
    return occ.astype(np.float32)
