from __future__ import annotations
import numpy as np

def zscore(X: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True) + eps
    return (X - mu) / sd

def smooth_1d(x: np.ndarray, win: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if win <= 1:
        return x
    k = np.ones(int(win), dtype=float) / float(win)
    return np.convolve(x, k, mode="same")
