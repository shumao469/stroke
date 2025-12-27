from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .segmentation import SegmentSpec, segment_by_time
from .preprocess import zscore
from .hmm import fit_hmm_gaussian, decode_hmm
from .manifold import run_pca, run_tsne
from .plotting import plot_state_sequence, plot_pca_scatter, plot_tsne_scatter

def build_parser():
    p = argparse.ArgumentParser(prog="xfastsort2-dynamics", description="HMM/PCA/tSNE for LFP or spike-derived features.")
    p.add_argument("--features-npy", required=True, help="(T,F) features array saved as .npy")
    p.add_argument("--fs", type=float, default=None, help="sampling rate for time axis, if no time file is provided")
    p.add_argument("--time-npy", default=None, help="(T,) time in seconds; if omitted, uses arange(T)/fs")
    p.add_argument("--outdir", required=True)

    p.add_argument("--stim-onset", type=float, default=0.0)
    p.add_argument("--pre-dur", type=float, default=60.0)
    p.add_argument("--stim-dur", type=float, default=180.0)
    p.add_argument("--post-dur", type=float, default=600.0)

    p.add_argument("--n-states", type=int, default=3)
    p.add_argument("--pca-dim", type=int, default=3)
    p.add_argument("--tsne-perplexity", type=float, default=30.0)
    return p

def main():
    args = build_parser().parse_args()
    X = np.load(args.features_npy)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if args.time_npy:
        t = np.load(args.time_npy).reshape(-1)
    else:
        if args.fs is None:
            raise SystemExit("Provide either --time-npy or --fs to build a time axis.")
        t = np.arange(X.shape[0], dtype=float) / float(args.fs)

    spec = SegmentSpec(stim_onset=args.stim_onset, pre_dur=args.pre_dur, stim_dur=args.stim_dur, post_dur=args.post_dur)
    seg = segment_by_time(t, X, spec)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # concatenate in order pre+stim+post for a single HMM fit
    t_all = np.concatenate([seg["pre"][0], seg["stim"][0], seg["post"][0]])
    X_all = np.concatenate([seg["pre"][1], seg["stim"][1], seg["post"][1]])

    X_all_z = zscore(X_all, axis=0)

    model = fit_hmm_gaussian(X_all_z, n_states=args.n_states)
    res = decode_hmm(model, X_all_z)

    np.save(outdir/"hmm_state_seq.npy", res.state_seq)
    np.save(outdir/"hmm_state_prob.npy", res.state_prob)

    plot_state_sequence(t_all, res.state_seq, outdir/"hmm_state_seq.png")

    pca = run_pca(X_all_z, n_components=args.pca_dim)
    np.save(outdir/"pca_embedding.npy", pca.Z)
    plot_pca_scatter(pca.Z, res.state_seq, outdir/"pca_pc1_pc2.png", title="PCA (colored by HMM state)")

    Zt = run_tsne(X_all_z, perplexity=args.tsne_perplexity)
    np.save(outdir/"tsne_embedding.npy", Zt)
    plot_tsne_scatter(Zt, res.state_seq, outdir/"tsne.png", title="t-SNE (colored by HMM state)")

    meta = {
        "features_npy": str(args.features_npy),
        "outdir": str(outdir),
        "segment_spec": vars(spec),
        "n_states": args.n_states,
        "pca_dim": args.pca_dim,
        "tsne_perplexity": args.tsne_perplexity,
    }
    (outdir/"run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
