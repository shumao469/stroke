from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

REQUIRED_COLUMNS_DEFAULT = (
    "age",
    "stroke_onset_months",
    "pre_fma",
    "pre_mbi",
    "stroke_type",
    "post_fma",
)

def load_clinical_csv(path: str | Path) -> pd.DataFrame:
    """Load clinical/demographic data from a CSV file."""
    return pd.read_csv(Path(path))

def validate_schema(
    df: pd.DataFrame,
    required_columns: Sequence[str] = REQUIRED_COLUMNS_DEFAULT,
    stroke_type_values: Tuple[str, str] = ("ischemic", "hemorrhagic"),
    strict: bool = False,
) -> None:
    """Validate required columns; optionally validate stroke_type values."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if strict:
        bad = df[~df["stroke_type"].astype(str).str.lower().isin(stroke_type_values)]
        if len(bad) > 0:
            raise ValueError(
                f"stroke_type contains unexpected values. Allowed: {stroke_type_values}. "
                f"Examples: {bad['stroke_type'].head(5).tolist()}"
            )
