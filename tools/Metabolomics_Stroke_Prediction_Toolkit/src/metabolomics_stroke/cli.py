from __future__ import annotations
import argparse
import os
import pandas as pd
import numpy as np
from .io import load_excel_tables, merge_abundance_tables
from .taxonomy import significant_subset, nested_donut_sunburst, sankey_alluvial
from .network import top_metabolites, spearman_corr, choose_threshold, build_graph, plot_network, plot_chord_class_density
from .predict_demo import demo_binary_prediction

DEFAULT_HSNC_SHEET = "HS-vs-NC_差异表达矩阵(未筛选)"
DEFAULT_ZSHS_SHEET = "ZS-vs-HS_差异表达矩阵(未筛选)"
DEFAULT_MET_COL = "Metabolites"

def cmd_load():
    ap = argparse.ArgumentParser(description="Load and merge abundance tables.")
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--hsnc_sheet", default=DEFAULT_HSNC_SHEET)
    ap.add_argument("--zshs_sheet", default=DEFAULT_ZSHS_SHEET)
    args = ap.parse_args()

    hsnc, zshs = load_excel_tables(args.excel, args.hsnc_sheet, args.zshs_sheet)
    mat = merge_abundance_tables(hsnc, zshs, metabolite_col=DEFAULT_MET_COL)
    os.makedirs(args.out, exist_ok=True)
    mat.to_csv(os.path.join(args.out, "abundance_merged.csv"))
    print("[OK] saved abundance_merged.csv")

def cmd_diff():
    print("Placeholder: add your full differential analysis scripts here if needed.")

def cmd_taxonomy():
    ap = argparse.ArgumentParser(description="Chemical taxonomy shift (Sunburst + Alluvial).")
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out", default="outputs/taxonomy")
    ap.add_argument("--p_thr", type=float, default=0.05)
    ap.add_argument("--vip_thr", type=float, default=1.0)
    ap.add_argument("--top_subclass", type=int, default=20)
    ap.add_argument("--sheet_hsnc", default="volcano-HS-vs-NC")
    ap.add_argument("--sheet_zshs", default="volcano-ZS-vs-HS")
    args = ap.parse_args()

    df1 = pd.read_excel(args.excel, sheet_name=args.sheet_hsnc)
    df2 = pd.read_excel(args.excel, sheet_name=args.sheet_zshs)

    sig1 = significant_subset(df1, p_thr=args.p_thr, vip_thr=args.vip_thr)
    sig2 = significant_subset(df2, p_thr=args.p_thr, vip_thr=args.vip_thr)

    os.makedirs(args.out, exist_ok=True)

    nested_donut_sunburst(sig1, "HS vs NC | Chemical taxonomy (sig set)", "HSvsNC_sunburst", args.out, with_text=True)
    nested_donut_sunburst(sig1, "HS vs NC | Chemical taxonomy (sig set)", "HSvsNC_sunburst", args.out, with_text=False)
    nested_donut_sunburst(sig2, "ZS vs HS | Chemical taxonomy (sig set)", "ZSvsHS_sunburst", args.out, with_text=True)
    nested_donut_sunburst(sig2, "ZS vs HS | Chemical taxonomy (sig set)", "ZSvsHS_sunburst", args.out, with_text=False)

    sankey_alluvial(sig1, "HS vs NC | SuperClass → SubClass → Regulation", "HSvsNC_sankey", args.out, with_text=True, top_k_sub=args.top_subclass)
    sankey_alluvial(sig1, "HS vs NC | SuperClass → SubClass → Regulation", "HSvsNC_sankey", args.out, with_text=False, top_k_sub=args.top_subclass)
    sankey_alluvial(sig2, "ZS vs HS | SuperClass → SubClass → Regulation", "ZSvsHS_sankey", args.out, with_text=True, top_k_sub=args.top_subclass)
    sankey_alluvial(sig2, "ZS vs HS | SuperClass → SubClass → Regulation", "ZSvsHS_sankey", args.out, with_text=False, top_k_sub=args.top_subclass)

    print(f"[OK] taxonomy figures saved: {args.out}")

def cmd_network():
    ap = argparse.ArgumentParser(description="Metabolite correlation network + chord-like density.")
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out", default="outputs/network")
    ap.add_argument("--top_n", type=int, default=150)
    ap.add_argument("--rank_by", default="VIP", choices=["VIP","p","|log2FC|"])
    ap.add_argument("--target_edges", type=int, default=650)
    ap.add_argument("--rho_min", type=float, default=0.70)
    ap.add_argument("--rho_max", type=float, default=0.88)
    ap.add_argument("--hsnc_sheet", default=DEFAULT_HSNC_SHEET)
    ap.add_argument("--zshs_sheet", default=DEFAULT_ZSHS_SHEET)
    ap.add_argument("--comparison", default="HSvsNC", choices=["HSvsNC","ZSvsHS"])
    args = ap.parse_args()

    hsnc, zshs = load_excel_tables(args.excel, args.hsnc_sheet, args.zshs_sheet)
    df = hsnc if args.comparison == "HSvsNC" else zshs

    top = top_metabolites(df, n=args.top_n, rank_by=args.rank_by)
    sample_cols = [c for c in df.columns if str(c).strip().startswith(("NC","HS","ZS","QC"))]

    X = top.set_index("Metabolites")[sample_cols].apply(pd.to_numeric, errors="coerce").T.dropna(axis=1, how="any")
    corr = spearman_corr(X)

    tax = top.set_index("Metabolites")
    class_map = tax["Class"].fillna("Unclassified").astype(str).to_dict() if "Class" in tax.columns else {}
    vip_map = pd.to_numeric(tax["VIP"], errors="coerce").to_dict() if "VIP" in tax.columns else {}

    M = corr.values
    iu = np.triu_indices_from(M, k=1)
    abs_vals = np.abs(M[iu])
    thr = choose_threshold(abs_vals, target_edges=args.target_edges, min_thr=args.rho_min, max_thr=args.rho_max)

    G = build_graph(corr, class_map, vip_map, thr=thr)
    os.makedirs(args.out, exist_ok=True)

    prefix = f"{args.comparison}_thr{thr:.2f}".replace(".", "p")
    plot_network(G, f"{args.comparison} | Spearman network (Top {X.shape[1]} by {args.rank_by})", thr,
                 prefix + "_network", args.out, with_text=True)
    plot_network(G, f"{args.comparison} | Spearman network (Top {X.shape[1]} by {args.rank_by})", thr,
                 prefix + "_network", args.out, with_text=False)

    plot_chord_class_density(G, f"{args.comparison} | Class-to-Class density", thr,
                             prefix + "_chord", args.out, with_text=True)
    plot_chord_class_density(G, f"{args.comparison} | Class-to-Class density", thr,
                             prefix + "_chord", args.out, with_text=False)

    print(f"[OK] network/chord figures saved: {args.out}")

def cmd_predict_demo():
    ap = argparse.ArgumentParser(description="Prediction core figures (DEMO only).")
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out", default="outputs/predict_demo")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--hsnc_sheet", default=DEFAULT_HSNC_SHEET)
    ap.add_argument("--zshs_sheet", default=DEFAULT_ZSHS_SHEET)
    args = ap.parse_args()

    hsnc, zshs = load_excel_tables(args.excel, args.hsnc_sheet, args.zshs_sheet)
    mat = merge_abundance_tables(hsnc, zshs, metabolite_col=DEFAULT_MET_COL)

    def top_by_vip(df, k):
        d = df[["Metabolites","VIP"]].copy()
        d["VIP"] = pd.to_numeric(d["VIP"], errors="coerce")
        d = d.dropna(subset=["VIP"]).sort_values("VIP", ascending=False)
        return d["Metabolites"].drop_duplicates().head(k).tolist()

    os.makedirs(args.out, exist_ok=True)

    # HS vs NC
    feats = top_by_vip(hsnc, args.top_k)
    X = mat.loc[feats].T
    X = X.loc[[i for i in X.index if str(i).startswith(("HS","NC"))]].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="any")
    demo_binary_prediction(X, "HS", args.out, "HS_vs_NC")

    # HS vs ZS
    feats = top_by_vip(zshs, args.top_k)
    X = mat.loc[feats].T
    X = X.loc[[i for i in X.index if str(i).startswith(("HS","ZS"))]].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="any")
    demo_binary_prediction(X, "HS", args.out, "HS_vs_ZS")

    # ZS vs NC by |log2FC|
    nc = [c for c in mat.columns if str(c).startswith("NC")]
    zs = [c for c in mat.columns if str(c).startswith("ZS")]
    mu_nc = mat[nc].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    mu_zs = mat[zs].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    eps = 1e-6
    log2fc = np.log2((mu_zs+eps)/(mu_nc+eps))
    feats = log2fc.abs().sort_values(ascending=False).head(args.top_k).index.tolist()

    X = mat.loc[feats].T
    X = X.loc[[i for i in X.index if str(i).startswith(("ZS","NC"))]].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="any")
    demo_binary_prediction(X, "ZS", args.out, "ZS_vs_NC")

    print(f"[OK] demo prediction figures saved: {args.out}")
