from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.io import loadmat as _loadmat, savemat as _savemat
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def read_txt_to_numpy(
    file_path: str | Path,
    data_cols: Tuple[int, ...] = (1,2,3,4,5,6,7,8,11),
    comment: str = '%',
    delimiter: Optional[str] = None,
) -> np.ndarray:
    """
    Read a .txt EEG export into a NumPy array.

    Parameters
    ----------
    file_path:
        Path to text file.
    data_cols:
        0-based column indices to extract (default matches MATLAB: [2:9,12] -> (1..8,11)).
        The last selected column is often a trigger/marker channel.
    comment:
        Comment line prefix to skip (default '%').
    delimiter:
        Delimiter for pandas. None lets pandas infer (works for whitespace/CSV).

    Returns
    -------
    arr: np.ndarray
        shape (n_samples, n_selected_cols)
    """
    file_path = Path(file_path)
    df = pd.read_csv(
        file_path,
        comment=comment,
        sep=delimiter if delimiter is not None else None,
        engine="python",
        header=None
    )
    return df.iloc[:, list(data_cols)].to_numpy()

def zero_out_trigger_codes(arr: np.ndarray, trigger_col: int = -1, codes=(0.512, 1.024)) -> np.ndarray:
    """Set rows with specific trigger codes to zero (port of raw.m cleanup blocks)."""
    out = arr.copy()
    trig = out[:, trigger_col]
    mask = np.zeros_like(trig, dtype=bool)
    for c in codes:
        mask |= np.isclose(trig, c)
    out[mask, :] = 0
    return out

def save_mat(file_path: str | Path, **variables: Any) -> None:
    """Save variables to a MATLAB .mat file (v7.3 not used)."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _savemat(str(file_path), variables)

def load_mat(file_path: str | Path) -> Dict[str, Any]:
    """Load a MATLAB .mat file into a dict, dropping MATLAB metadata keys."""
    d = _loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in d.items() if not k.startswith("__")}
