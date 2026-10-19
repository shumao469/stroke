# 10 Hz example (TI): How to cut pre/stim/post, align, and run HMM/PCA/tSNE

This walkthrough uses the included example feature file:

- `examples/data/TI_10Hz_3min_rec10min_pre_cleaned_PAC.npy`

> If your file is a **feature time series** (e.g., PAC, band-power, spike-rate), you can run analyses directly.
> If your file is raw LFP, first compute features in sliding windows (see next section).

---

## 1) Understand the recording timeline

A typical 10 Hz TI session is:

- **pre:** 60 s baseline (no stimulation)
- **stim:** 180 s (3 min stimulation)
- **post:** 600 s (10 min after stim)

Total = 60 + 180 + 600 = **840 s**.

If your recording starts earlier or ends later, adjust durations.

---

## 2) Build the time axis and align to stim onset

### Case A: you already have aligned features
If your feature array has one row per time bin, you need a matching `t_sec`:

- If you computed features with window length `W` and step `S`,
  the canonical time for bin *i* is:

  `t[i] = t0 + i * S + W/2`

where `t0` is the start time (seconds).

### Case B: TTL-based alignment (recommended)
If you have a TTL rising edge sample index `s_onset`:

- `stim_onset = s_onset / fs`
- shift: `t_aligned = t_raw - stim_onset`
- then stim onset is exactly 0 s

---

## 3) How to cut pre/stim/post (recommended rule)

After alignment (`stim_onset = 0`):

- pre:  [-60, 0) seconds
- stim: [0, 180) seconds
- post: [180, 780) seconds

In code we use onset+durations:

- `pre_dur = 60`
- `stim_dur = 180`
- `post_dur = 600`

---

## 4) Running HMM/PCA/tSNE from CLI (fastest)

From repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

Then run:

```bash
xfastsort2-dynamics \
  --features-npy examples/data/TI_10Hz_3min_rec10min_pre_cleaned_PAC.npy \
  --fs 1 \
  --outdir outputs/10hz_demo \
  --stim-onset 0 \
  --pre-dur 60 --stim-dur 180 --post-dur 600 \
  --n-states 3 \
  --pca-dim 3 \
  --tsne-perplexity 30
```

### Important: what is `--fs` here?
For **feature time series**, `fs` means **feature sampling rate (bins per second)**, not raw LFP sampling rate.

Example:
- If you computed 1 feature per second → `fs = 1`
- If step is 0.5 s → `fs = 2`

If you have a separate time vector, pass `--time-npy`.

Outputs:
- `hmm_state_seq.npy/.png` (state over time)
- `pca_embedding.npy` + `pca_pc1_pc2.png`
- `tsne_embedding.npy` + `tsne.png`
- `run_meta.json`

---

## 5) Feature engineering for raw LFP or spikes (recommended)

### LFP features (sliding windows)
Window `W = 2 s`, step `S = 0.2 s` is a good default.

Per window compute:
- band power (delta/theta/alpha/beta/gamma)
- coherence (optional)
- PAC metrics (optional)

The result is `X` with shape `(T_bins, F_features)`.

### Spike features (sliding windows)
Per window compute:
- firing rate (Hz)
- burst index
- spike-LFP phase locking (optional)

These can be concatenated with LFP features.

Then run z-scoring before HMM.

---

## 6) Interpreting HMM + manifold results

- HMM gives a discrete **state sequence**; check:
  - whether stim period occupies distinct states from pre/post
  - whether transitions occur near onset/offset
- PCA/tSNE plots:
  - points colored by HMM state should form separable clusters
  - you can also color by segment (pre/stim/post)

For manuscript-ready results, report:
- number of states, training set, feature set, windowing params
- stability across random seeds and across animals/sessions
