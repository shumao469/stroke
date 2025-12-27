from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from .io import read_txt_to_numpy, save_mat, load_mat, zero_out_trigger_codes
from .preprocess import epoch_from_trigger_channel
from .timefreq import compute_ersp_morlet, save_ersp_outputs
from .erd_ers import compute_subject_metrics
from .stats_viz import compute_significance_matrices, plot_threshold_heatmaps

def cmd_txt2mat(args):
    arr = read_txt_to_numpy(args.input, data_cols=tuple(args.cols))
    if args.zero_codes:
        arr = zero_out_trigger_codes(arr, trigger_col=args.trigger_col, codes=tuple(args.zero_codes))
    save_mat(args.output, blinkData=arr)
    print(f"Saved: {args.output}")

def cmd_ersp(args):
    d = load_mat(args.input_mat)
    if args.var not in d:
        raise KeyError(f"Variable '{args.var}' not found in {args.input_mat}. Keys={list(d.keys())}")
    blink = np.asarray(d[args.var], dtype=float)  # (n_samples, n_cols)
    data = blink.T  # (n_channels, n_samples)
    ch_names = args.ch_names.split(",")
    if len(ch_names) != data.shape[0]:
        raise ValueError(f"ch_names count ({len(ch_names)}) must match data channels ({data.shape[0]})")

    epochs = epoch_from_trigger_channel(
        data=data,
        sfreq=args.sfreq,
        ch_names=ch_names,
        trigger_ch=args.trigger_ch,
        threshold=args.threshold,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=(args.bmin, args.bmax),
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        reref=args.reref,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    freqs = np.linspace(args.fmin, args.fmax, int(args.fmax - args.fmin) + 1)

    ep = epochs.get_data()  # (n_epochs, n_channels, n_times)
    for ch in range(args.n_eeg_ch):
        power, times_ms = compute_ersp_morlet(
            ep[:, ch, :],
            sfreq=args.sfreq,
            freqs=freqs,
            baseline=(args.bmin, args.bmax),
            mode="logratio",
        )
        save_ersp_outputs(out_dir, ch + 1, power, times_ms, freqs, vmin=args.vmin, vmax=args.vmax)

    eegdata = np.transpose(ep[:, :args.n_eeg_ch, :], (1, 2, 0))  # (ch, time, epoch)
    save_mat(out_dir / "eegdata_epoched.mat", eegdata=eegdata)
    print(f"Saved ERSP outputs to {out_dir}")

def cmd_erd_ers(args):
    d = load_mat(args.eegdata_mat)
    eeg = np.asarray(d[args.var], dtype=float)
    out = compute_subject_metrics(eeg, sfreq=args.sfreq, stim_side=args.stim_side)
    save_mat(args.output_mat, **out)
    print(f"Saved ERD/ERS metrics to {args.output_mat}")

def cmd_stats(args):
    mats = compute_significance_matrices(
        args.results_before,
        args.results_after,
        args.li_before,
        args.li_after,
        p_threshold=args.p_threshold,
        d_threshold=args.d_threshold,
    )
    plot_threshold_heatmaps(
        mats["sig_diff"],
        mats["effect_d"],
        mats["li_sig"],
        mats["li_d"],
        out_png=args.out_png,
    )
    print(f"Saved heatmaps: {args.out_png}")

def build_parser():
    p = argparse.ArgumentParser(prog="rtms-eeg", description="Task-state rTMS-EEG analysis utilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("txt2mat", help="Convert raw .txt exports to .mat (blinkData variable).")
    p1.add_argument("--input", required=True)
    p1.add_argument("--output", required=True)
    p1.add_argument("--cols", nargs="+", type=int, default=[1,2,3,4,5,6,7,8,11], help="0-based columns to keep")
    p1.add_argument("--trigger-col", type=int, default=-1, help="Index in kept columns for trigger")
    p1.add_argument("--zero-codes", nargs="*", type=float, default=[], help="Trigger codes to zero-out")
    p1.set_defaults(func=cmd_txt2mat)

    p2 = sub.add_parser("ersp", help="Epoch using trigger channel and compute ERSP (Morlet).")
    p2.add_argument("--input-mat", required=True)
    p2.add_argument("--var", default="blinkData")
    p2.add_argument("--sfreq", type=float, default=500.0)
    p2.add_argument("--ch-names", default="CP3,FC3,TP7,FT7,CP4,FC4,TP8,FT8,TRIG",
                    help="Comma-separated channel names, last should be trigger")
    p2.add_argument("--trigger-ch", default="TRIG")
    p2.add_argument("--threshold", type=float, default=None)
    p2.add_argument("--tmin", type=float, default=-2.0)
    p2.add_argument("--tmax", type=float, default=5.0)
    p2.add_argument("--bmin", type=float, default=-2.0)
    p2.add_argument("--bmax", type=float, default=0.0)
    p2.add_argument("--l-freq", type=float, default=0.5)
    p2.add_argument("--h-freq", type=float, default=40.0)
    p2.add_argument("--reref", default="average")
    p2.add_argument("--n-eeg-ch", type=int, default=8, help="Number of EEG channels (excluding trigger)")
    p2.add_argument("--fmin", type=float, default=1.0)
    p2.add_argument("--fmax", type=float, default=40.0)
    p2.add_argument("--vmin", type=float, default=-5.0)
    p2.add_argument("--vmax", type=float, default=5.0)
    p2.add_argument("--out-dir", required=True)
    p2.set_defaults(func=cmd_ersp)

    p3 = sub.add_parser("erd-ers", help="Compute ERD/ERS and LI from epoched eegdata (.mat).")
    p3.add_argument("--eegdata-mat", required=True)
    p3.add_argument("--var", default="eegdata")
    p3.add_argument("--sfreq", type=float, default=500.0)
    p3.add_argument("--stim-side", choices=["left", "right"], default="left")
    p3.add_argument("--output-mat", required=True)
    p3.set_defaults(func=cmd_erd_ers)

    p4 = sub.add_parser("stats", help="Generate significance/effect-size heatmaps from exported Excel.")
    p4.add_argument("--results-before", required=True)
    p4.add_argument("--results-after", required=True)
    p4.add_argument("--li-before", required=True)
    p4.add_argument("--li-after", required=True)
    p4.add_argument("--p-threshold", type=float, default=0.05)
    p4.add_argument("--d-threshold", type=float, default=0.5)
    p4.add_argument("--out-png", required=True)
    p4.set_defaults(func=cmd_stats)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
