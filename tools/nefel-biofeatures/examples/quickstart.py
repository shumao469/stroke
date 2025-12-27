import argparse
import cv2
import pandas as pd

from nefel.feature_extraction import split_channels, quantify_cd31, quantify_iba1, quantify_claudin5, quantify_gap43


def main():
    p = argparse.ArgumentParser(description="NEFEL Biofeature Extraction quickstart")
    p.add_argument("--image", required=True, help="Path to an input image (e.g., .jpg/.png)")
    p.add_argument("--out", default="metrics.csv", help="Output CSV path")
    args = p.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    ch = split_channels(bgr)

    # The notebook functions generally return a dict-like record
    rows = []
    rows.append(quantify_cd31(ch["rgb"], prefix="sample"))
    rows.append(quantify_iba1(ch["rgb"], prefix="sample"))
    rows.append(quantify_claudin5(ch["rgb"], prefix="sample"))
    rows.append(quantify_gap43(ch["rgb"], prefix="sample"))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
