# MATLAB → Python mapping

This document explains how each MATLAB script is mapped to Python modules/functions.

## raw.m
- MATLAB: readtable -> extract columns [2:9,12], save blinkData
- Python: rtms_eeg_closedloop.io.read_txt_to_numpy (default data_cols=(1..8,11) 0-based)
- Trigger cleanup:
  - MATLAB: blinkData(blinkData(:, 9) == 0.512, :) = 0;
  - Python: rtms_eeg_closedloop.io.zero_out_trigger_codes

## eeglabtask.m
- Import .mat -> MNE RawArray, filter 0.5–40 Hz, average reference
- pop_chanevent rising edge detection -> epoch_from_trigger_channel
- pop_epoch [-2 5] -> tmin=-2, tmax=5
- pop_rmbase [-2000 0] -> baseline=(-2, 0)
- newtimef -> compute_ersp_morlet (Morlet wavelets + logratio baseline)

## task.m
- Band power via Welch PSD (baseline vs task windows)
- Python: compute_erd_ers_ratio, compute_subject_metrics
- LI: -(mean(ipsi)-mean(contra))/(|mean(ipsi)|+|mean(contra)|)

## ptest1219.m
- Normality: KS test to N(0,1)
- Paired test: ttest_rel if normal else Wilcoxon signed-rank
- Effect size: pooled SD Cohen's d + sign-consistency rule
- Python: compute_significance_matrices + plot_threshold_heatmaps
