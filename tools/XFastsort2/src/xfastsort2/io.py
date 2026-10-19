from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

def load_csv_trace(
    csv_path: Union[str, Path],
    time_col: str = "Filtered (1) Timestamps",
    value_col: str = "Filtered (1) Values",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single-channel continuous trace from a CSV.

    Parameters
    ----------
    csv_path:
        Path to CSV (often exported from acquisition software).
    time_col, value_col:
        Column names for timestamps (seconds) and values.

    Returns
    -------
    t_sec: (N,) float64
    x: (N,) float32
    """
    df = pd.read_csv(Path(csv_path))
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"CSV missing required columns. Need time_col='{time_col}', value_col='{value_col}'.\n"
            f"Found columns: {list(df.columns)[:30]}"
        )
    t = df[time_col].to_numpy(dtype=np.float64)
    x = df[value_col].to_numpy(dtype=np.float32)
    return t, x

def load_binary_memmap(
    bin_path: Union[str, Path],
    n_channels: int,
    dtype: str = "int16",
    offset_bytes: int = 0,
) -> np.memmap:
    """Memory-map an interleaved multichannel binary file (channel-major interleaving).

    Layout assumed: [sample0_ch0, sample0_ch1, ..., sample0_ch{C-1}, sample1_ch0, ...]

    Returns a memmap view shaped (n_samples, n_channels).
    """
    bin_path = Path(bin_path)
    itemsize = np.dtype(dtype).itemsize
    file_bytes = bin_path.stat().st_size - int(offset_bytes)
    if file_bytes <= 0:
        raise ValueError("Binary file appears empty or offset too large.")
    n_total = file_bytes // itemsize
    if n_total % n_channels != 0:
        raise ValueError(
            f"File size not divisible by n_channels. total_items={n_total}, n_channels={n_channels}"
        )
    n_samples = n_total // n_channels
    mm = np.memmap(bin_path, dtype=dtype, mode="r", offset=offset_bytes, shape=(n_samples, n_channels))
    return mm
