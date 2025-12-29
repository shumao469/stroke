from __future__ import annotations
import re
import pandas as pd
from .utils import find_sample_cols

def load_excel_tables(excel_path: str, sheet_hsnc: str, sheet_zshs: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hsnc = pd.read_excel(excel_path, sheet_name=sheet_hsnc)
    zshs = pd.read_excel(excel_path, sheet_name=sheet_zshs)
    return hsnc, zshs

def merge_abundance_tables(hsnc: pd.DataFrame, zshs: pd.DataFrame, metabolite_col: str = "Metabolites") -> pd.DataFrame:
    s1 = find_sample_cols(hsnc.columns)
    s2 = find_sample_cols(zshs.columns)

    m1 = hsnc[[metabolite_col] + s1].copy()
    m2 = zshs[[metabolite_col] + s2].copy()
    merged = pd.merge(m1, m2, on=metabolite_col, how="inner", suffixes=("_hsnc","_zshs"))

    def resolve_group(df: pd.DataFrame, prefix: str) -> dict[str, pd.Series]:
        cols = [c for c in df.columns if re.match(rf"^{prefix}\d+(_hsnc|_zshs)?$", str(c).strip())]
        base = sorted({re.sub(r"(_hsnc|_zshs)$","", str(c)) for c in cols})
        out: dict[str, pd.Series] = {}
        for b in base:
            c0, c1, c2 = b, b+"_hsnc", b+"_zshs"
            if c1 in df.columns and c2 in df.columns:
                out[b] = pd.concat([pd.to_numeric(df[c1], errors="coerce"),
                                    pd.to_numeric(df[c2], errors="coerce")], axis=1).mean(axis=1)
            elif c0 in df.columns:
                out[b] = pd.to_numeric(df[c0], errors="coerce")
            elif c1 in df.columns:
                out[b] = pd.to_numeric(df[c1], errors="coerce")
            elif c2 in df.columns:
                out[b] = pd.to_numeric(df[c2], errors="coerce")
        return out

    out = pd.DataFrame({metabolite_col: merged[metabolite_col]})
    for c in [c for c in merged.columns if re.match(r"^NC\d+$", str(c).strip())]:
        out[str(c).strip()] = pd.to_numeric(merged[c], errors="coerce")

    for prefix in ["HS","ZS","QC"]:
        for k,v in resolve_group(merged, prefix).items():
            out[k] = v

    return out.drop_duplicates(subset=[metabolite_col]).set_index(metabolite_col)
