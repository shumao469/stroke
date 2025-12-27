# Kilosort4 quick guide (for XFastsort2 users)

Kilosort is a state-of-the-art GPU-accelerated spike sorting pipeline developed by MouseLand
and widely used for high-channel-count extracellular recordings.

> If you use Kilosort1–4, please cite:  
> Pachitariu, M., Sridhar, S., Pennington, J., & Stringer, C. (2024). *Spike sorting with Kilosort4*. **Nature Methods**.

This guide summarizes practical installation and usage steps (please also consult the upstream Kilosort documentation).

---

## System requirements (rule-of-thumb)

- Windows / Linux 64-bit supported
- GPU strongly recommended
- **≥ 8 GB** GPU RAM (12 GB+ recommended)
- SSD for data and outputs improves speed

---

## Installation (conda recommended)

1) Install Anaconda/Miniconda  
2) Create environment:

```bash
conda create -n kilosort python=3.11
conda activate kilosort
```

3) Install Kilosort:

```bash
python -m pip install kilosort[gui]
```

4) Install CUDA-enabled PyTorch (example for CUDA 11.8):

```bash
pip uninstall -y torch
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

> Choose the CUDA build matching your GPU driver/toolkit.
> Some newer GPUs may work best with newer CUDA builds (nightly builds can be used if needed).

---

## Running Kilosort4

### GUI mode (recommended for first-time setup)

```bash
python -m kilosort
```

Then:
- select binary file path + results directory
- select probe configuration (MAT recommended)
- click **LOAD** then **Run**
- outputs are compatible with **Phy** (manual curation GUI)

### Phy curation

In the Kilosort result directory, run:

```bash
phy template-gui params.py
```

---

## Notes about older versions

The upstream repo reports known bugs affecting spike detection at batch boundaries in older versions (2/2.5/3).
Kilosort4 is recommended for new use.

---

## Using Kilosort4 from XFastsort2

If you installed `kilosort`, you can call:

```bash
xfastsort2 kilosort4 --bin your.bin --probe your_probe.mat --outdir results/ --fs 20000 --n-channels 384
```

If your installed Kilosort version exposes a different python API, use the GUI or consult upstream docs.
