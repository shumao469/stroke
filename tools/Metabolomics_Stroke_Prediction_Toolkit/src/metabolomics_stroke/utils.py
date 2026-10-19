from __future__ import annotations
import re
from typing import Iterable, List

SAMPLE_RE = re.compile(r"^(QC|NC|HS|ZS)\d+$", re.IGNORECASE)

def find_sample_cols(columns: Iterable[str]) -> List[str]:
    return [c for c in columns if SAMPLE_RE.match(str(c).strip())]
