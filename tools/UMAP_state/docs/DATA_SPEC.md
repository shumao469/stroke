# DATA_SPEC — Feature input format for UMAP State

UMAP State operates on **feature time series**, not raw LFP.

## Minimal input
- `.npy` with shape `(T, F)` (or `(T,)` for single feature)
  - `T`: number of feature bins
  - `F`: number of features per bin
- Optional `.npy` time vector `(T,)` in seconds.

If time is not provided, you must supply `--fs`, where **fs is feature sampling rate** (bins/second):
`t = arange(T) / fs`

## Recommended feature construction
### LFP-derived features (sliding windows)
Compute features in sliding windows, e.g.:
- window length W = 2 s
- step S = 0.2 s

Per window:
- band power: delta/theta/alpha/beta/gamma
- PAC metrics
- coherence or PLV (optional)

### Spike-derived features (sliding windows)
Per window:
- firing rate
- burst index
- spike-field coupling (optional)

Then concatenate features → `X(T,F)` and z-score before embedding.

## Saving
- `np.save("features.npy", X.astype(np.float32))`
- `np.save("time.npy", t.astype(np.float32))` (recommended)
