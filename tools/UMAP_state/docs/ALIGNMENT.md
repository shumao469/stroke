# Alignment (stim onset = 0 s) and segmentation

UMAP state analysis is only interpretable when sessions are aligned to a shared reference event.

## Recommended alignment
Define the time axis so that **stimulation onset = 0 s**:

- If you have a TTL rising edge sample index `s_onset`:
  - `stim_onset = s_onset / fs_raw`
  - `t_aligned = t_raw - stim_onset`

- If your features are computed in windows:
  - Let window length be `W` seconds, step be `S` seconds.
  - The bin time can be defined as `t[i] = t0 + i*S + W/2`.
  - Apply the same shift so that the first stimulation bin aligns to 0.

## Typical TI 10 Hz protocol (example)
After alignment:

- pre:  [-60, 0) s
- stim: [0, 180) s   (3 min)
- post: [180, 780) s (10 min)

These are implemented in the CLI with:
`--stim-onset 0 --pre-dur 60 --stim-dur 180 --post-dur 600`
