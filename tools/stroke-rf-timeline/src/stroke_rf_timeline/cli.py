from __future__ import annotations

import argparse
from pathlib import Path
import json

import pandas as pd

from .data import load_clinical_csv, validate_schema
from .pipeline import RFConfig, train_validate, save_model

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stroke-rf",
        description="Random forest regression for clinical outcome prediction (ΔFMA).",
    )
    p.add_argument("--csv", required=True, help="Input CSV file with clinical/demographic data.")
    p.add_argument("--outdir", required=True, help="Output directory for model and predictions.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--mice-iter", type=int, default=5, help="Iterations for IterativeImputer (MICE-like).")
    return p

def main():
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_clinical_csv(args.csv)
    validate_schema(df, strict=False)

    cfg = RFConfig(random_state=args.seed, test_size=args.test_size, mice_max_iter=args.mice_iter)
    gs, metrics, preds = train_validate(df, config=cfg)

    preds_path = outdir / "validation_predictions.csv"
    preds.to_csv(preds_path, index=False)

    model_path = outdir / "rf_gridsearch.joblib"
    save_model(gs, model_path)

    metrics_path = outdir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved:")
    print(f"  Model:   {model_path}")
    print(f"  Preds:   {preds_path}")
    print(f"  Metrics: {metrics_path}")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
