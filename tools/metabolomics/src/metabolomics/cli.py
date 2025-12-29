"""
Command-line entrypoints for the Metabolomics pipeline.

This is a scaffold. Current workflows live in ./scripts/.
"""
from __future__ import annotations
import argparse
from .utils import ensure_dir

def main():
    ap = argparse.ArgumentParser(prog="metabolomics")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("qc", help="Generate QC figures (PCA/RSD/correlation)")
    p.add_argument("--xlsx", required=True)
    p.add_argument("--sheet", default="缺失值数据矩阵")
    p.add_argument("--out_dir", default="QC_Figures")

    p2 = sub.add_parser("predict", help="Run repeated CV + OOF prediction & figures")
    p2.add_argument("--xlsx", required=True)
    p2.add_argument("--sheet", default="缺失值数据矩阵")
    p2.add_argument("--out_dir", default="out_predict")

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    raise SystemExit(
        "CLI scaffold only. Run scripts in ./scripts/ or wire your functions into src/metabolomics/*."
    )

if __name__ == "__main__":
    main()
