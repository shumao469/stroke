from __future__ import annotations
import argparse, json
from pathlib import Path

from .pipeline import sort_csv_trace_cpu, batch_sort_csv_cpu
from .kilosort_wrapper import run_kilosort4

def build_parser():
    p = argparse.ArgumentParser(prog="xfastsort2", description="XFastsort2: spike sorting utilities + Kilosort4 helpers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("cpu-sort", help="CPU spike sorting for a single-channel CSV trace.")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--outdir", required=True)
    p1.add_argument("--fs", type=float, default=20000.0)
    p1.add_argument("--n-clusters", type=int, default=3)
    p1.add_argument("--thresh-sd", type=float, default=4.0)
    p1.add_argument("--polarity", default="negative")

    p2 = sub.add_parser("batch-cpu-sort", help="Batch CPU sorting over a folder of CSV traces.")
    p2.add_argument("--root", required=True)
    p2.add_argument("--out-root", required=True)
    p2.add_argument("--fs", type=float, default=20000.0)
    p2.add_argument("--pattern", default=".csv")

    p3 = sub.add_parser("kilosort4", help="Run Kilosort4 (requires kilosort installed).")
    p3.add_argument("--bin", required=True, help="binary file path")
    p3.add_argument("--probe", required=True, help="probe config path (MAT/PRB)")
    p3.add_argument("--outdir", required=True, help="output results directory")
    p3.add_argument("--fs", type=float, required=True, help="sample rate")
    p3.add_argument("--n-channels", type=int, required=True)
    p3.add_argument("--dtype", default="int16")
    p3.add_argument("--device", default="cuda")

    return p

def main():
    args = build_parser().parse_args()
    if args.cmd == "cpu-sort":
        res = sort_csv_trace_cpu(
            args.csv, args.outdir, fs=args.fs, n_clusters=args.n_clusters, thresh_sd=args.thresh_sd, polarity=args.polarity
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "batch-cpu-sort":
        res = batch_sort_csv_cpu(args.root, args.out_root, fs=args.fs, pattern=args.pattern)
        print(json.dumps(res, indent=2))
    elif args.cmd == "kilosort4":
        out = run_kilosort4(
            args.bin, args.probe, args.outdir,
            sample_rate=args.fs, n_channels=args.n_channels, dtype=args.dtype, device=args.device
        )
        print(f"Saved Kilosort4 results to: {out}")
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
