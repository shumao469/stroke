from __future__ import annotations
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm

def top_metabolites(df: pd.DataFrame, n: int = 150, rank_by: str = "VIP", p_col: str = "p-value") -> pd.DataFrame:
    d = df.copy()
    if rank_by.upper() == "VIP":
        d["VIP"] = pd.to_numeric(d["VIP"], errors="coerce")
        d = d.sort_values("VIP", ascending=False)
    elif rank_by.lower() in ["p","pvalue","p-value"]:
        d[p_col] = pd.to_numeric(d[p_col], errors="coerce")
        d = d.sort_values(p_col, ascending=True)
    elif rank_by.lower() in ["|log2fc|","abslog2fc","abs_log2fc"]:
        d["log2FoldChange"] = pd.to_numeric(d["log2FoldChange"], errors="coerce")
        d = d.assign(_abs=d["log2FoldChange"].abs()).sort_values("_abs", ascending=False)
    else:
        raise ValueError("rank_by must be one of VIP, p, |log2FC|")
    return d.drop_duplicates(subset=["Metabolites"]).head(n).copy()

def spearman_corr(X: pd.DataFrame) -> pd.DataFrame:
    return X.T.corr(method="spearman")

def choose_threshold(abs_vals: np.ndarray, target_edges: int = 650,
                     min_thr: float = 0.70, max_thr: float = 0.88) -> float:
    v = abs_vals[np.isfinite(abs_vals)]
    v = v[(v < 0.999999) & (v >= 0)]
    if len(v) == 0:
        return 0.80
    s = np.sort(v)[::-1]
    k = min(max(target_edges, 1), len(s))
    thr = float(s[k-1])
    return float(np.clip(thr, min_thr, max_thr))

def build_graph(corr: pd.DataFrame, class_map: dict, vip_map: dict, thr: float) -> nx.Graph:
    mets = corr.index.tolist()
    G = nx.Graph()
    for m in mets:
        G.add_node(m, Class=class_map.get(m,"Unclassified"), VIP=float(vip_map.get(m, np.nan)))
    mat = corr.values
    n = len(mets)
    for i in range(n):
        for j in range(i+1, n):
            rho = mat[i,j]
            if np.isfinite(rho) and abs(rho) >= thr:
                G.add_edge(mets[i], mets[j], rho=float(rho), w=float(abs(rho)))
    return G

def _palette(classes):
    cmap = cm.get_cmap("tab20", max(len(classes), 1))
    return {c: cmap(i) for i,c in enumerate(classes)}

def plot_network(G: nx.Graph, title: str, thr: float, out_prefix: str, out_dir: str, with_text: bool = True) -> None:
    classes = sorted({G.nodes[n].get("Class","Unclassified") for n in G.nodes})
    pal = _palette(classes)
    node_colors = [pal[G.nodes[n].get("Class","Unclassified")] for n in G.nodes]
    vips = np.array([G.nodes[n].get("VIP", np.nan) for n in G.nodes], dtype=float)
    vips = np.nan_to_num(vips, nan=np.nanmedian(vips) if np.isfinite(vips).any() else 1.0)
    vmin, vmax = float(vips.min()), float(vips.max())
    sizes = (60 + 340*(vips-vmin)/(vmax-vmin)) if (vmax-vmin)>1e-9 else np.full_like(vips, 120.0)

    pos = nx.spring_layout(G, seed=7, k=1.2/math.sqrt(max(G.number_of_nodes(),1)))

    weights = [G.edges[e]["w"] for e in G.edges] if G.number_of_edges()>0 else [0.0]
    wmin, wmax = min(weights), max(weights)
    ewidth = [0.6 for _ in G.edges] if (wmax-wmin)<1e-9 else [0.3+2.6*(w-wmin)/(wmax-wmin) for w in weights]

    fig = plt.figure(figsize=(10.5,10.5))
    ax = plt.gca(); ax.axis("off"); ax.set_title(title, fontsize=14, pad=14)
    nx.draw_networkx_edges(G, pos, ax=ax, width=ewidth, alpha=0.35)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=sizes,
                           linewidths=0.8, edgecolors="black", alpha=0.95)
    if with_text:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    ax.text(0.99, 0.01, f"Edges: |rho|≥{thr:.2f} | Spearman (visualization only)",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_{'text' if with_text else 'notext'}.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

def plot_chord_class_density(G: nx.Graph, title: str, thr: float, out_prefix: str, out_dir: str, with_text: bool = True) -> None:
    nodes = list(G.nodes)
    cls = {n: G.nodes[n].get("Class","Unclassified") for n in nodes}
    classes = sorted(set(cls.values()))
    class_nodes = {c:[n for n in nodes if cls[n]==c] for c in classes}
    sizes = {c:len(v) for c,v in class_nodes.items()}

    edge_counts = {(a,b):0 for a in classes for b in classes if a<=b}
    for u,v in G.edges:
        cu, cv = cls[u], cls[v]
        a,b = (cu,cv) if cu<=cv else (cv,cu)
        edge_counts[(a,b)] += 1

    density = {}
    for a in classes:
        for b in classes:
            if a<=b:
                na, nb = sizes[a], sizes[b]
                possible = na*nb if a!=b else na*(na-1)/2
                density[(a,b)] = 0.0 if possible<=0 else edge_counts[(a,b)]/possible

    pairs = [((a,b),v) for (a,b),v in density.items() if a!=b and v>0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pairs[:min(40, len(pairs))]

    total = sum(sizes.values()) if sum(sizes.values())>0 else 1
    gaps = 0.03
    span = 2*math.pi - gaps*len(classes)
    arcs = {}
    start = math.pi/2
    for c in classes:
        ang = span*(sizes[c]/total)
        arcs[c] = (start, start-ang)
        start = start-ang-gaps
    midang = {c:(arcs[c][0]+arcs[c][1])/2 for c in classes}

    pal = _palette(classes)

    fig = plt.figure(figsize=(10,10))
    ax = plt.gca(); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=14, pad=14)

    R=1.0
    for c in classes:
        a0,a1 = arcs[c]
        t = np.linspace(a1,a0,120)
        ax.plot(R*np.cos(t), R*np.sin(t), linewidth=10, color=pal[c], solid_capstyle="butt")
        if with_text:
            ang = midang[c]
            lx, ly = 1.12*np.cos(ang), 1.12*np.sin(ang)
            rot = np.degrees(ang)-90
            ha = "left" if np.cos(ang)>0 else "right"
            ax.text(lx, ly, c, rotation=rot, ha=ha, va="center", fontsize=9)

    dens_vals = [v for _,v in top_pairs]
    dmin, dmax = (min(dens_vals), max(dens_vals)) if dens_vals else (0.0, 1.0)

    def arc_point(c):
        ang = midang[c]
        return 0.98*np.cos(ang), 0.98*np.sin(ang), ang

    for (a,b), dens in top_pairs:
        x0,y0,ang0 = arc_point(a)
        x1,y1,ang1 = arc_point(b)
        cx0,cy0 = 0.55*np.cos(ang0), 0.55*np.sin(ang0)
        cx1,cy1 = 0.55*np.cos(ang1), 0.55*np.sin(ang1)
        lw = 2.0 if (dmax-dmin)<1e-12 else (0.6 + 6.0*(dens-dmin)/(dmax-dmin))
        ax.plot([x0,cx0,cx1,x1], [y0,cy0,cy1,y1], alpha=0.45, linewidth=lw, color="black")

    ax.text(0.99, 0.01, f"Chord-like density from edges (|rho|≥{thr:.2f}); tiny n → visualization only",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_{'text' if with_text else 'notext'}.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
