from __future__ import annotations
import os
import re
from typing import List
import pandas as pd

SAMPLE_COL_RE = re.compile(r"^(NC|HS|ZS|QC)\d+$", re.IGNORECASE)

def find_sample_columns(df: pd.DataFrame) -> List[str]:
    """Identify sample columns such as NC1/HS1/ZS1/QC1..."""
    return [c for c in df.columns if SAMPLE_COL_RE.match(str(c).strip())]

def infer_group_from_sample_id(sample_id: str) -> str:
    m = re.match(r"^(NC|HS|ZS|QC)", str(sample_id).strip(), flags=re.IGNORECASE)
    return m.group(1).upper() if m else "UNK"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
