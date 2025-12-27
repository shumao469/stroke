from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _ensure(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def plot_umap(Z: np.ndarray, labels: np.ndarray, out_png: str | Path, title: str = "UMAP state space"):
    out_png = Path(out_png); _ensure(out_png)
    plt.figure(figsize=(5,4))
    plt.scatter(Z[:,0], Z[:,1], s=6, c=labels, alpha=0.85)
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def plot_umap_segments(Z: np.ndarray, seg_id: np.ndarray, out_png: str | Path, title: str = "UMAP colored by segment"):
    out_png = Path(out_png); _ensure(out_png)
    plt.figure(figsize=(5,4))
    plt.scatter(Z[:,0], Z[:,1], s=6, c=seg_id, alpha=0.85)
    plt.xlabel("UMAP1"); plt.ylabel("UMAP2"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def plot_state_sequence(t: np.ndarray, labels: np.ndarray, out_png: str | Path, title: str = "State sequence"):
    out_png = Path(out_png); _ensure(out_png)
    plt.figure(figsize=(10,2.5))
    plt.step(t, labels, where="post")
    plt.xlabel("Time (s)"); plt.ylabel("State"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def plot_transition_matrix(P: np.ndarray, out_png: str | Path, title: str = "Transition matrix"):
    out_png = Path(out_png); _ensure(out_png)
    plt.figure(figsize=(4.5,4))
    plt.imshow(P, aspect="auto")
    plt.colorbar(label="P(next|current)")
    plt.xlabel("Next state"); plt.ylabel("Current state"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def plot_occupancy(occ: np.ndarray, out_png: str | Path, title: str = "State occupancy"):
    out_png = Path(out_png); _ensure(out_png)
    plt.figure(figsize=(5,3))
    plt.bar(np.arange(len(occ)), occ)
    plt.xlabel("State"); plt.ylabel("Fraction"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
