# Magnetic NeuroRing rTMS-EEG closed loop control

Python implementation of **task or rest-state rTMS–EEG** analysis utilities, ported from MATLAB/EEGLAB scripts:

- `raw.m` — TXT → MAT + trigger cleanup  
- `eeglabtask.m` — filter → reref → epoch → baseline → ERSP per channel  
- `task.m` — ERD/ERS across frequency bands + Laterality Index (LI)  
- `ptest1219.m` — paired statistics + effect size heatmaps with thresholds  

Designed for easy GitHub sharing and reproducible analysis.

## Paper / citation
If you find this code useful, please cite:

```bibtex
@article{NEFEL2025@ISTBI_Fudan Magnetic NeuroRing: A portable adaptive brain-computer interface for real-time transcranial magnetic stimulation in post-stroke motor rehabilitation
  journal={npj Biomedical Innovations},
  year={2025},
  doi={10.1038/s44385-025-00055-5}
}
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Command-line entrypoint:

```bash
rtms-eeg --help
```

## Data assumptions (important)

### Raw TXT export
Your MATLAB `raw.m` indicates each `.txt` contains:
- Column 1: time
- Columns 2–9: EEG channels (8 channels)
- Column 12: trigger/marker channel  
Python defaults follow this mapping (`data_cols=(1..8,11)` using 0-based indices).

### MAT variables
- `referdata_*.mat` contains `blinkData` with shape `(n_samples, n_cols)`
- `eegdata_epoched.mat` contains `eegdata` with shape `(n_channels, n_times, n_epochs)`

## Usage (CLI)

### 1) Convert TXT → MAT (port of `raw.m`)

```bash
rtms-eeg txt2mat --input sub1/task1.txt --output sub1/referdata_task1.mat
```

Optional: zero-out specific trigger codes (matches your MATLAB blocks using 0.512 / 1.024):

```bash
rtms-eeg txt2mat --input sub1/task1.txt --output sub1/referdata_task1.mat --zero-codes 0.512 1.024
```

### 2) Epoch + compute ERSP (port of `eeglabtask.m`)

```bash
rtms-eeg ersp --input-mat sub1/referdata_task1.mat --out-dir outputs/sub1_task1
```

Notes:
- Trigger detection uses a robust threshold (`median + 3*MAD`). If no triggers are found, supply `--threshold`.
- Filtering uses MNE FIR (`0.5–40 Hz` by default).
- ERSP uses Morlet wavelets + baseline correction (`logratio`) to approximate EEGLAB `newtimef`.

### 3) Compute ERD/ERS + LI (port of `task.m`)

```bash
rtms-eeg erd-ers --eegdata-mat outputs/sub1_task1/eegdata_epoched.mat --stim-side left --output-mat outputs/sub1_task1/erd_ers.mat
```

### 4) Stats & effect-size heatmaps (port of `ptest1219.m`)

```bash
rtms-eeg stats   --results-before results_before2.xlsx   --results-after  results_after2.xlsx   --li-before results_li_before2.xlsx   --li-after  results_li_after2.xlsx   --p-threshold 0.05   --d-threshold 0.5   --out-png stats_heatmaps.png
```

## Module layout
- `rtms_eeg_closedloop/io.py` — txt/mat IO utilities
- `rtms_eeg_closedloop/preprocess.py` — filtering, re-referencing, epoching from trigger channel
- `rtms_eeg_closedloop/timefreq.py` — ERSP computation + saving
- `rtms_eeg_closedloop/erd_ers.py` — ERD/ERS + LI metrics
- `rtms_eeg_closedloop/stats_viz.py` — paired tests + heatmap plots
- `rtms_eeg_closedloop/cli.py` — CLI (`rtms-eeg`)

## Contact
If you have any questions, please feel free to contact 📩 **shumaoxu@fudan.edu.cn**
