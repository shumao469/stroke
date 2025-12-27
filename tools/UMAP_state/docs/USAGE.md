# How to use UMAP State (step-by-step)

## 1) Install

```bash
pip install -r requirements.txt
pip install -e .
```

Optional (recommended for state clustering):
- `hdbscan` (if build fails on Windows, use `--cluster kmeans`)

Optional GPU:
- RAPIDS cuML UMAP can be used with `--gpu` flag if `cuml` is installed.

## 2) Run on a 10 Hz session (pre/stim/post)

Assume you already computed features and aligned stim onset to 0.

```bash
umap-state \
  --features path/to/features.npy \
  --time path/to/time.npy \
  --outdir outputs/umap_10hz \
  --stim-onset 0 \
  --pre-dur 60 --stim-dur 180 --post-dur 600 \
  --n-neighbors 30 --min-dist 0.1 \
  --cluster hdbscan --min-cluster-size 50
```

If you do not have time.npy:
- pass `--fs` (feature sampling rate, bins/sec), e.g. 5 for 0.2 s step.

```bash
umap-state --features features.npy --fs 5 --outdir outputs/umap_10hz ...
```

## 3) Outputs

`outdir` contains:
- `umap_embedding.npy` (T,2)
- `state_labels.npy` (T,)
- `state_sequence.png`
- `umap_states.png` (colored by state)
- `umap_segments.png` (colored by pre/stim/post)
- `transition_matrix.npy/.png`
- `state_occupancy.npy/.png`
- `run_meta.json`

## 4) Interpretation

- Check whether stim points occupy distinct regions/states vs pre/post.
- Inspect state transitions near onset/offset.
- Compare occupancy changes (pre vs stim vs post) by running per-segment embeddings or by segmenting labels.
