from pathlib import Path
import pandas as pd
import re

root = Path("/mnt/h/Data/Yuchun-yanshi")
out = root / "qc_audit_report.xlsx"

def iter_tables(root):
    exts = {".xlsx", ".xls", ".csv", ".tsv", ".txt"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def read_table(p):
    suf = p.suffix.lower()
    if suf in [".csv"]:
        return {"__single__": pd.read_csv(p)}
    if suf in [".tsv", ".txt"]:
        return {"__single__": pd.read_csv(p, sep="\t")}
    if suf in [".xlsx"]:
        x = pd.ExcelFile(p, engine="openpyxl")
        return {s: x.parse(s) for s in x.sheet_names[:20]}  # 防止超大文件
    if suf in [".xls"]:
        # 很多“伪xls”其实是tsv，先按tsv读
        try:
            return {"__single__": pd.read_csv(p, sep="\t", engine="python")}
        except Exception:
            return {"__single__": pd.read_csv(p, engine="python")}
    return {}

def looks_like_sample_col(c):
    c = str(c)
    return bool(re.match(r"^(QC|HS|NC|ZS)\d+$", c, flags=re.I))

hits = []
for p in iter_tables(root):
    try:
        sheets = read_table(p)
    except Exception:
        continue
    for sheet, df in sheets.items():
        # 1) 列名是否含QC
        cols = [str(c) for c in df.columns]
        col_qc = [c for c in cols if re.search(r"\bQC\b", c, flags=re.I) or c.lower().startswith("qc")]
        # 2) 单元格是否含QC（只抽样前200行前50列，避免太慢）
        sub = df.iloc[:200, :50].astype(str)
        cell_has_qc = sub.apply(lambda s: s.str.contains(r"\bQC\b", case=False, na=False)).any().any()
        # 3) 是否存在类似 QC1/HS1/NC1/ZS1 的样本列
        sample_like = [c for c in cols if looks_like_sample_col(c)]
        if col_qc or cell_has_qc or sample_like:
            hits.append({
                "file": str(p),
                "sheet": sheet,
                "n_rows": df.shape[0],
                "n_cols": df.shape[1],
                "qc_cols": ", ".join(col_qc[:20]),
                "sample_like_cols": ", ".join(sample_like[:30]),
                "cell_contains_QC": bool(cell_has_qc),
            })

report = pd.DataFrame(hits).sort_values(["file","sheet"])
with pd.ExcelWriter(out, engine="openpyxl") as w:
    report.to_excel(w, index=False, sheet_name="QC_hits")
print("Saved:", out, "rows:", len(report))
