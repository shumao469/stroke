# Alignment & segmentation (Pre/Stim/Post)

## Why alignment matters
In TI experiments, HMM/PCA/tSNE is meaningful only if each trial/recording is aligned to the same reference event:
- **stimulus onset** (e.g., TI start TTL)
- **behavioral trigger** (tail pinch/heat)
- **task cue**

Recommended: define time axis such that **stim onset = 0 s**.
Then:
- `pre` is negative time (e.g., -60 to 0 s)
- `stim` is 0 to +180 s (3 min)
- `post` is +180 to +780 s (10 min post)

## Common alignment sources

### A) Hardware TTL (best)
- acquisition system logs a digital channel
- detect rising edge → `stim_onset_sample`
- convert to seconds: `stim_onset = stim_onset_sample / fs`

### B) Recorded stimulation waveform
- detect envelope or beat signal
- locate onset by threshold/correlation

### C) Manual annotation
- use time stamp from experiment log, but record uncertainty

## After alignment
Build `t_sec` and set `stim_onset = 0` by shifting:

```
t_aligned = t_raw - stim_onset
```

Then you can segment by durations using `SegmentSpec(stim_onset=0, pre_dur=60, stim_dur=180, post_dur=600)`.
