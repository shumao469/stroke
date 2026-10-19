from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def significant_subset(df: pd.DataFrame, p_col: str = "p-value", vip_col: str = "VIP",
                       p_thr: float = 0.05, vip_thr: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    for c in ["Super Class","Class","Sub Class","Regulation"]:
        if c in out.columns:
            out[c] = out[c].fillna("Unclassified").astype(str)
    out[p_col] = pd.to_numeric(out[p_col], errors="coerce")
    out[vip_col] = pd.to_numeric(out[vip_col], errors="coerce")
    sig = out[(out[p_col] < p_thr) & (out[vip_col] >= vip_thr)].copy()
    if "Regulation" in sig.columns:
        sig["Regulation"] = sig["Regulation"].astype(str).str.strip().str.capitalize()
        sig.loc[~sig["Regulation"].isin(["Up","Down"]), "Regulation"] = "NA"
    return sig

def nested_donut_sunburst(sig_df: pd.DataFrame, title: str, out_prefix: str, out_dir: str,
                          with_text: bool = True, max_sub_labels: int = 15) -> None:
    d = sig_df.copy()
    d["n"] = 1
    super_counts = d.groupby("Super Class")["n"].sum().sort_values(ascending=False)
    class_counts = d.groupby(["Super Class","Class"])["n"].sum().sort_values(ascending=False)
    sub_counts = d.groupby(["Super Class","Class","Sub Class"])["n"].sum().sort_values(ascending=False)

    ring1_labels = super_counts.index.tolist()
    ring1_sizes = super_counts.values.tolist()

    ring2_sizes, ring2_labels = [], []
    ring3_sizes, ring3_labels = [], []
    for sc in ring1_labels:
        classes = class_counts.loc[sc].sort_values(ascending=False)
        for cl, ncl in classes.items():
            ring2_sizes.append(ncl); ring2_labels.append(str(cl))
            subs = sub_counts.loc[(sc, cl)].sort_values(ascending=False)
            for subc, nsub in subs.items():
                ring3_sizes.append(nsub); ring3_labels.append(str(subc))

    total = int(sum(ring1_sizes))
    fig = plt.figure(figsize=(9,9))
    ax = plt.gca()
    ax.set(aspect="equal")
    ax.set_title(title, fontsize=14, pad=18)

    labels1 = ring1_labels if with_text else [""]*len(ring1_labels)
    ax.pie(ring1_sizes, radius=1.0, labels=labels1, labeldistance=0.70,
           wedgeprops=dict(width=0.25, edgecolor="white"))

    if with_text:
        labels2 = [lab if (s/total) >= 0.06 else "" for s, lab in zip(ring2_sizes, ring2_labels)]
    else:
        labels2 = [""]*len(ring2_labels)
    ax.pie(ring2_sizes, radius=0.75, labels=labels2, labeldistance=0.75,
           wedgeprops=dict(width=0.25, edgecolor="white"))

    if with_text:
        top_subs = pd.Series(ring3_sizes, index=ring3_labels).sort_values(ascending=False).head(max_sub_labels).index
        labels3 = [lab if lab in set(top_subs) else "" for lab in ring3_labels]
    else:
        labels3 = [""]*len(ring3_labels)
    ax.pie(ring3_sizes, radius=0.50, labels=labels3, labeldistance=0.78,
           wedgeprops=dict(width=0.25, edgecolor="white"))

    centre = plt.Circle((0,0), 0.20, fc="white")
    ax.add_artist(centre)
    ax.text(0, 0, f"Sig\nn={total}", ha="center", va="center", fontsize=12)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_{'text' if with_text else 'notext'}.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)

def sankey_alluvial(sig_df: pd.DataFrame, title: str, out_prefix: str, out_dir: str,
                    with_text: bool = True, top_k_sub: int = 20) -> None:
    d = sig_df.copy()
    d["weight"] = 1.0
    flows = d.groupby(["Super Class","Sub Class","Regulation"], as_index=False)["weight"].sum()
    flows = flows[flows["Regulation"].isin(["Up","Down"])].copy()
    if flows.empty:
        return

    totals = flows.groupby("Sub Class")["weight"].sum().sort_values(ascending=False)
    keep = set(totals.head(top_k_sub).index.tolist())
    flows["Sub Class"] = flows["Sub Class"].where(flows["Sub Class"].isin(keep), "Other")
    flows = flows.groupby(["Super Class","Sub Class","Regulation"], as_index=False)["weight"].sum()

    def stack(series, gap=0.012):
        usable = 1.0 - gap*(len(series)-1)
        pos = {}
        y = 1.0
        for name, w in series.items():
            h = usable * (w/series.sum())
            pos[name] = (y-h, y)
            y = y-h-gap
        return pos

    left_tot = flows.groupby("Super Class")["weight"].sum().sort_values(ascending=False)
    mid_tot  = flows.groupby("Sub Class")["weight"].sum().sort_values(ascending=False)
    right_tot= flows.groupby("Regulation")["weight"].sum().reindex(["Up","Down"]).fillna(0)

    left_pos = stack(left_tot)
    mid_pos  = stack(mid_tot)
    right_pos= stack(right_tot)

    left_cur  = {k:left_pos[k][0] for k in left_pos}
    mid_in    = {k:mid_pos[k][0] for k in mid_pos}
    mid_out   = {k:mid_pos[k][0] for k in mid_pos}
    right_cur = {k:right_pos[k][0] for k in right_pos}

    xL, xM, xR = 0.10, 0.50, 0.90
    node_w = 0.04

    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)

    import matplotlib.patches as patches
    def draw_nodes(pos, x):
        for name,(y0,y1) in pos.items():
            rect = patches.Rectangle((x-node_w/2,y0), node_w, y1-y0, facecolor="none", linewidth=1.0)
            ax.add_patch(rect)
            if with_text:
                ax.text(x, (y0+y1)/2, str(name), ha="center", va="center", fontsize=8)
    draw_nodes(left_pos, xL); draw_nodes(mid_pos, xM); draw_nodes(right_pos, xR)

    def bezier(x0,y0,x1,y1,lw):
        c1 = (x0 + (x1-x0)*0.35, y0)
        c2 = (x0 + (x1-x0)*0.65, y1)
        verts = [(x0,y0), c1, c2, (x1,y1)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor="none", linewidth=lw, alpha=0.45))

    flows = flows.sort_values(["Super Class","Sub Class","Regulation"])
    max_w = float(flows["weight"].max())

    for _, r in flows.iterrows():
        w = float(r["weight"])
        sc, sub, reg = str(r["Super Class"]), str(r["Sub Class"]), str(r["Regulation"])

        yL0 = left_cur[sc]; yL1 = yL0 + (left_pos[sc][1]-left_pos[sc][0])*(w/left_tot[sc]); left_cur[sc]=yL1
        yM0 = mid_in[sub];  yM1 = yM0 + (mid_pos[sub][1]-mid_pos[sub][0])*(w/mid_tot[sub]); mid_in[sub]=yM1
        yM2 = mid_out[sub]; yM3 = yM2 + (mid_pos[sub][1]-mid_pos[sub][0])*(w/mid_tot[sub]); mid_out[sub]=yM3
        yR0 = right_cur[reg]; yR1 = yR0 + (right_pos[reg][1]-right_pos[reg][0])*(w/right_tot[reg]); right_cur[reg]=yR1

        lw = 0.5 + 6.0*(w/max_w)
        bezier(xL+node_w/2, (yL0+yL1)/2, xM-node_w/2, (yM0+yM1)/2, lw)
        bezier(xM+node_w/2, (yM2+yM3)/2, xR-node_w/2, (yR0+yR1)/2, lw)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ["png","pdf"]:
        fig.savefig(os.path.join(out_dir, f"{out_prefix}_{'text' if with_text else 'notext'}.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
