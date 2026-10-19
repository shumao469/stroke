from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import DBSPointCloudDataset
from .model import DBSPointNetPP
from .train import train_one_epoch, evaluate
from .build_pointcloud import fuse_pointcloud_kdtree

def build_parser():
    p = argparse.ArgumentParser(prog="dbs-pointnetpp", description="PointNet++ DBS efficacy prediction.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train PointNet++ on .npz point clouds.")
    p_train.add_argument("--data-root", required=True, help="Folder containing per-patient .npz files.")
    p_train.add_argument("--outdir", required=True, help="Output directory.")
    p_train.add_argument("--num-points", type=int, default=20000)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--lambda-reg", type=float, default=0.5)
    p_train.add_argument("--val-ratio", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p_fuse = sub.add_parser("fuse", help="Fuse anatomy/electrode/E-field arrays into a unified point cloud.")
    p_fuse.add_argument("--anatomy-npy", required=True, help="(Na,3) anatomy xyz .npy")
    p_fuse.add_argument("--electrode-npy", required=True, help="(Ne,3) electrode xyz .npy")
    p_fuse.add_argument("--efield-xyz-npy", required=True, help="(Mf,3) efield xyz .npy")
    p_fuse.add_argument("--efield-val-npy", required=True, help="(Mf,) efield magnitude .npy")
    p_fuse.add_argument("--out-npz", required=True, help="Output .npz with 'points'.")
    p_fuse.add_argument("--k", type=int, default=32)

    return p

def cmd_train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds = DBSPointCloudDataset(args.data_root, num_points=args.num_points, normalize=True)
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # infer in_channels from first sample
    p0, _, _ = ds[0]
    in_ch = int(p0.shape[1])

    model = DBSPointNetPP(in_channels=in_ch, n_classes=1).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = -1e9
    history = []

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, args.device, lambda_reg=args.lambda_reg)
        va = evaluate(model, val_loader, args.device)
        row = {"epoch": ep, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)

        score = va.get("r2", float("-inf"))
        if score > best:
            best = score
            torch.save(model.state_dict(), outdir / "best_model.pt")

        print(row)

    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (outdir / "best_score.json").write_text(json.dumps({"best_r2": best}, indent=2), encoding="utf-8")

def cmd_fuse(args):
    anatomy = np.load(args.anatomy_npy)
    electrode = np.load(args.electrode_npy)
    e_xyz = np.load(args.efield_xyz_npy)
    e_val = np.load(args.efield_val_npy)

    points = fuse_pointcloud_kdtree(anatomy, electrode, e_xyz, e_val, k=args.k)
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, points=points, y_reg=np.float32(0.0), y_cls=np.int64(-1))
    print(f"Saved fused point cloud: {args.out_npz} (points shape={points.shape})")

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "fuse":
        cmd_fuse(args)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
