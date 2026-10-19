# Quickstart (edit paths before running)
from pathlib import Path
import numpy as np
from rtms_eeg_closedloop.io import read_txt_to_numpy, save_mat
from rtms_eeg_closedloop.preprocess import epoch_from_trigger_channel
from rtms_eeg_closedloop.timefreq import compute_ersp_morlet, save_ersp_outputs

txt_path = Path("sub1/task1.txt")
out_dir = Path("outputs/sub1_task1")
out_dir.mkdir(parents=True, exist_ok=True)

arr = read_txt_to_numpy(txt_path)
save_mat(out_dir / "referdata.mat", blinkData=arr)

data = arr.T
ch_names = ["CP3","FC3","TP7","FT7","CP4","FC4","TP8","FT8","TRIG"]

epochs = epoch_from_trigger_channel(data=data, sfreq=500.0, ch_names=ch_names, trigger_ch="TRIG",
                                   tmin=-2.0, tmax=5.0, baseline=(-2.0, 0.0))

freqs = np.arange(1, 41)
ep = epochs.get_data()
for ch in range(8):
    power, times_ms = compute_ersp_morlet(ep[:, ch, :], sfreq=500.0, freqs=freqs, baseline=(-2.0, 0.0))
    save_ersp_outputs(out_dir / "ersp", ch+1, power, times_ms, freqs)
