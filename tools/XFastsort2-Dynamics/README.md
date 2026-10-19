# XFastsort2-Dynamics (HMM / PCA / t-SNE for TI spike & LFP features)
LFP-spike sorting analyses via XFAstsort2.5: https://xfastsort2-5.streamlit.app/

This repository provides a **reproducible analysis pipeline** for extracting low-dimensional structure and latent states
from **LFP or spike-derived time-series features** (e.g., band power, PAC features, firing rate).

It is designed to complement spike sorting (e.g., XFastsort2 / Kilosort) and focus on **state dynamics** across
**pre / stimulation / post** epochs.

---

## What you get

- **Segmentation utilities** for pre/stim/post
- **Alignment guidance** (stim onset = 0 s recommended)
- **Gaussian HMM** state inference (via `hmmlearn`)
- **PCA** and **t-SNE** embeddings (via scikit-learn)
- A clean CLI: `xfastsort2-dynamics` to run end-to-end
- A **10 Hz TI example walkthrough** (docs/TENHZ_WALKTHROUGH.md)

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quick start (10 Hz example)

Run on the included example feature file:

```bash
xfastsort2-dynamics \
  --features-npy examples/data/TI_10Hz_3min_rec10min_pre_cleaned_PAC.npy \
  --fs 1 \
  --outdir outputs/10hz_demo \
  --stim-onset 0 \
  --pre-dur 60 --stim-dur 180 --post-dur 600 \
  --n-states 3
```

See:
- docs/ALIGNMENT.md
- docs/TENHZ_WALKTHROUGH.md

---

## Data interface (minimal)

Input features must be saved as `.npy`:

- `features.npy`: shape `(T, F)` (or `(T,)` for 1D feature)
- optional `time.npy`: shape `(T,)` time in seconds

If you do not provide `time.npy`, the CLI will generate:
- `t = arange(T) / fs`
where `fs` is the **feature sampling rate (bins per second)**.

---

## Outputs

For each run, `outdir` contains:

- `hmm_state_seq.npy` and `hmm_state_seq.png`
- `hmm_state_prob.npy`
- `pca_embedding.npy` and `pca_pc1_pc2.png`
- `tsne_embedding.npy` and `tsne.png`
- `run_meta.json`

---

## Citation

If you use this code, please cite your paper / preprint here (replace the placeholder):

```bibtex
@article{your2026nefel,
  title={...},
  author={...},
  journal={...},
  year={2026}
}
```

---

## Contact

For questions and contributions, please open a GitHub issue or contact the maintainers.
