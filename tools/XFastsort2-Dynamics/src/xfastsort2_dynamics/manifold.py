from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

@dataclass
class PCAResult:
    Z: np.ndarray               # (T, n_components)
    components: np.ndarray      # (n_components, F)
    explained_var: np.ndarray   # (n_components,)

def run_pca(X: np.ndarray, n_components: int = 3, random_state: int = 0) -> PCAResult:
    from sklearn.decomposition import PCA
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    pca = PCA(n_components=int(n_components), random_state=int(random_state))
    Z = pca.fit_transform(X)
    return PCAResult(Z=Z.astype(np.float32), components=pca.components_.astype(np.float32), explained_var=pca.explained_variance_ratio_.astype(np.float32))

def run_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    random_state: int = 0,
) -> np.ndarray:
    from sklearn.manifold import TSNE
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    tsne = TSNE(
        n_components=int(n_components),
        perplexity=float(perplexity),
        learning_rate=float(learning_rate),
        n_iter=int(n_iter),
        init="pca",
        random_state=int(random_state),
    )
    Z = tsne.fit_transform(X)
    return Z.astype(np.float32)
