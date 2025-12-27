from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

def load_features(features_path: Union[str, Path]) -> np.ndarray:
    """Load features from .npy or .npz.

    Supported:
      - .npy: array (T,F) or (T,)
      - .npz: must contain key 'X' or 'features'
    """
    p = Path(features_path)
    if p.suffix.lower() == ".npy":
        X = np.load(p)
    elif p.suffix.lower() == ".npz":
        z = np.load(p)
        if "X" in z:
            X = z["X"]
        elif "features" in z:
            X = z["features"]
        else:
            raise ValueError("NPZ must contain key 'X' or 'features'.")
    else:
        raise ValueError("features_path must be .npy or .npz")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def load_time(time_path: Union[str, Path]) -> np.ndarray:
    """Load time vector from .npy."""
    p = Path(time_path)
    t = np.load(p).reshape(-1)
    return t.astype(float)
