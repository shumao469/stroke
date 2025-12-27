from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def plot_state_sequence(t: np.ndarray, states: np.ndarray, out_png: str | Path, title: str = "HMM state sequence"):
    out_png = Path(out_png); ensure_dir(out_png)
    plt.figure(figsize=(10,2.5))
    plt.step(t, states, where="post")
    plt.xlabel("Time (s)"); plt.ylabel("State"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def plot_pca_scatter(Z: np.ndarray, labels: np.ndarray, out_png: str | Path, title: str = "PCA"):
    out_png = Path(out_png); ensure_dir(out_png)
    plt.figure(figsize=(5,4))
    plt.scatter(Z[:,0], Z[:,1], s=8, c=labels, alpha=0.8)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def plot_tsne_scatter(Z: np.ndarray, labels: np.ndarray, out_png: str | Path, title: str = "t-SNE"):
    out_png = Path(out_png); ensure_dir(out_png)
    plt.figure(figsize=(5,4))
    plt.scatter(Z[:,0], Z[:,1], s=8, c=labels, alpha=0.8)
    plt.xlabel("tSNE1"); plt.ylabel("tSNE2"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
