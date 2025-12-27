from pathlib import Path
import numpy as np
import pandas as pd

from xfastsort2.pipeline import sort_csv_trace_cpu

# Create a tiny synthetic demo trace (NOT physiological; for sanity-check only)
fs = 20000
t = np.arange(fs * 2) / fs
x = 0.05 * np.random.randn(t.size).astype(np.float32)
# add a few synthetic spikes
for s in [10000, 15000, 22000, 30000]:
    if s+20 < len(x):
        x[s:s+5] -= 0.8

df = pd.DataFrame({"Filtered (1) Timestamps": t, "Filtered (1) Values": x})
csv = Path(__file__).parent / "demo_trace.csv"
df.to_csv(csv, index=False)

outdir = Path(__file__).parent / "outputs_demo"
res = sort_csv_trace_cpu(csv, outdir, fs=fs, n_clusters=3)
print(res)
