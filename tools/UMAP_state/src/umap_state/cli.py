from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from .io import load_features, load_time
from .preprocess import zscore
from .segmentation import SegmentSpec, segment_by_time
from .embed import UMAPConfig, fit_umap
from .state import ClusterConfig, cluster_states, transition_matrix, state_occupancy
from .plotting import (
    plot_umap, plot_umap_segments, plot_state_sequence, plot_transition_matrix, plot_occupancy
)

def build_parser():
    p = argparse.ArgumentParser(prog="umap-state", description="UMAP state-space analysis for TI spike/LFP features.")
    p.add_argument("--features", required=True, help="features .npy or .npz (T,F)")
    p.add_argument("--time", default=None, help="time .npy (T,) in seconds; if omitted uses arange(T)/fs")
    p.add_argument("--fs", type=float, default=None, help="feature sampling rate (bins per second) if --time omitted")
    p.add_argument("--outdir", required=True)

    # segmentation (10Hz typical)
    p.add_argument("--stim-onset", type=float, default=0.0)
    p.add_argument("--pre-dur", type=float, default=60.0)
    p.add_argument("--stim-dur", type=float, default=180.0)
    p.add_argument("--post-dur", type=float, default=600.0)

    # UMAP
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--metric", default="euclidean")
    p.add_argument("--umap-dim", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", action="store_true", help="use cuML UMAP if available")

    # clustering
    p.add_argument("--cluster", choices=["hdbscan", "kmeans"], default="hdbscan")
    p.add_argument("--min-cluster-size", type=int, default=50)
    p.add_argument("--kmeans-k", type=int, default=5)

    return p

def main():
    args = build_parser().parse_args()
    X = load_features(args.features)
    if args.time:
        t = load_time(args.time)
    else:
        if args.fs is None:
            raise SystemExit("Provide --time or --fs (feature sampling rate).")
        t = np.arange(X.shape[0], dtype=float) / float(args.fs)

    spec = SegmentSpec(stim_onset=args.stim_onset, pre_dur=args.pre_dur, stim_dur=args.stim_dur, post_dur=args.post_dur)
    seg = segment_by_time(t, X, spec)

    # concatenate pre+stim+post for joint embedding
    t_all = np.concatenate([seg["pre"][0], seg["stim"][0], seg["post"][0]])
    X_all = np.concatenate([seg["pre"][1], seg["stim"][1], seg["post"][1]])

    # segment id for plotting: 0=pre,1=stim,2=post
    seg_id = np.concatenate([
        np.zeros(seg["pre"][0].shape[0], dtype=int),
        np.ones(seg["stim"][0].shape[0], dtype=int),
        np.full(seg["post"][0].shape[0], 2, dtype=int),
    ])

    Xz = zscore(X_all, axis=0)

    umcfg = UMAPConfig(
        n_components=args.umap_dim,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
        use_gpu=bool(args.gpu),
    )
    model, Z, backend = fit_umap(Xz, umcfg)

    clcfg = ClusterConfig(
        method=args.cluster,
        min_cluster_size=args.min_cluster_size,
        n_clusters=args.kmeans_k,
        random_state=args.seed,
    )
    labels = cluster_states(Z, clcfg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir/"umap_embedding.npy", Z)
    np.save(outdir/"state_labels.npy", labels)
    np.save(outdir/"segment_id.npy", seg_id)
    np.save(outdir/"time_sec.npy", t_all)

    P = transition_matrix(labels)
    occ = state_occupancy(labels)
    np.save(outdir/"transition_matrix.npy", P)
    np.save(outdir/"state_occupancy.npy", occ)

    plot_umap(Z, labels, outdir/"umap_states.png", title=f"UMAP states ({backend})")
    plot_umap_segments(Z, seg_id, outdir/"umap_segments.png")
    plot_state_sequence(t_all, labels, outdir/"state_sequence.png")
    if P.size:
        plot_transition_matrix(P, outdir/"transition_matrix.png")
    if occ.size:
        plot_occupancy(occ, outdir/"state_occupancy.png")

    meta = {
        "features": str(args.features),
        "time": str(args.time) if args.time else None,
        "fs": args.fs,
        "segment_spec": vars(spec),
        "umap": {
            "backend": backend,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "dim": args.umap_dim,
            "seed": args.seed,
            "gpu_requested": bool(args.gpu),
        },
        "clustering": {
            "method": args.cluster,
            "min_cluster_size": args.min_cluster_size,
            "kmeans_k": args.kmeans_k,
        }
    }
    (outdir/"run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
