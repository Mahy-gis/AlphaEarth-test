import argparse
import json
from pathlib import Path

import numpy as np


def load_colormap(colormap_path: Path) -> np.ndarray:
    """Load colormap.txt -> (N,3) float32 in [0,1]."""
    rows: list[tuple[float, float, float]] = []
    with colormap_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expected format: (r, g, b), one per line
            line = line.rstrip(",")
            if line.startswith("(") and line.endswith(")"):
                line = line[1:-1]
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) != 3:
                continue
            r, g, b = map(float, parts)
            rows.append((r, g, b))
    if not rows:
        raise ValueError(f"No valid colors parsed from {colormap_path}")
    return np.asarray(rows, dtype=np.float32)


def color_mask_to_labels(color_mask: np.ndarray, colormap: np.ndarray) -> np.ndarray:
    """Map (H,W,3) float color mask to (H,W) int class indices via nearest color.

    Args:
        color_mask: (H,W,3) in [0,1] or [0,255]
        colormap: (N,3) in [0,1]
    """
    if color_mask.ndim != 3 or color_mask.shape[-1] != 3:
        raise ValueError(f"Expected color_mask with shape (H,W,3), got {color_mask.shape}")

    cm = colormap.astype(np.float32)
    img = color_mask.astype(np.float32)

    # Normalize image to [0,1] if it looks like 0-255 RGB
    max_val = img.max()
    if max_val > 1.0 + 1e-3:
        img = img / 255.0

    H, W, _ = img.shape
    img_flat = img.reshape(-1, 3)  # (Npix,3)

    # Compute squared distances to each colormap entry
    diff = img_flat[:, None, :] - cm[None, :, :]  # (Npix,Ncolors,3)
    dist2 = np.sum(diff * diff, axis=-1)          # (Npix,Ncolors)
    idx = np.argmin(dist2, axis=1).astype(np.int64)

    return idx.reshape(H, W)


def convert_single_label_file(
    label_path: Path,
    colormap_path: Path,
    label_names_meta: np.ndarray | None,
    out_path: Path,
) -> None:
    if not label_path.exists():
        raise FileNotFoundError(f"Label npy not found: {label_path}")
    if not colormap_path.exists():
        raise FileNotFoundError(f"Colormap file not found: {colormap_path}")

    print(f"Loading labels from {label_path}...")
    raw = np.load(label_path)
    print(f"  raw labels shape: {raw.shape}, dtype={raw.dtype}")

    # Case 1: already small integer class IDs (H,W) in [0,19]
    if raw.ndim == 2:
        uniq = np.unique(raw)
        print(f"  raw unique values (truncated): {uniq[:20]}{'...' if uniq.size > 20 else ''}")

        if uniq.min() >= 0 and uniq.max() <= 19:
            labels = raw.astype(np.int64)
            print("Detected integer class ID mask (H,W) in [0,19], using as-is.")
            print(f"  unique class IDs: {np.unique(labels)}")
        else:
            # Likely a packed color image where each int encodes RGB as 24-bit.
            print(
                "Detected (H,W) integer mask with large values; "
                "treating as packed RGB (24-bit) and decoding to colors."
            )
            v = raw.astype(np.int64)
            r = (v >> 16) & 255
            g = (v >> 8) & 255
            b = v & 255
            color_img = np.stack([r, g, b], axis=-1).astype(np.uint8)  # (H,W,3)

            print(f"Loading colormap from {colormap_path}...")
            colormap = load_colormap(colormap_path)
            print(f"  colormap has {colormap.shape[0]} entries")

            print("Mapping packed colors to class IDs (0..N-1)...")
            labels = color_mask_to_labels(color_img, colormap)
            print(f"  labels shape: {labels.shape}, dtype={labels.dtype}")
            print(f"  unique class IDs: {np.unique(labels)}")
    else:
        # Case 2: explicit color mask (H,W,3) that must be mapped via colormap
        if raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError(
                f"Unsupported label_npy shape {raw.shape}; expected (H,W) or (H,W,3)."
            )

        print(f"Loading colormap from {colormap_path}...")
        colormap = load_colormap(colormap_path)
        print(f"  colormap has {colormap.shape[0]} entries")

        print("Mapping colors to class IDs (0..N-1)...")
        labels = color_mask_to_labels(raw, colormap)
        print(f"  labels shape: {labels.shape}, dtype={labels.dtype}")
        print(f"  unique class IDs: {np.unique(labels)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out_path}...")
    if out_path.suffix == ".npz":
        if label_names_meta is not None:
            np.savez(out_path, labels=labels.astype(np.int64), label_names=label_names_meta)
        else:
            np.savez(out_path, labels=labels.astype(np.int64))
    else:
        # For .npy or other, just save labels array
        np.save(out_path, labels.astype(np.int64))

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert label npy files to integer label masks (H,W) with class IDs 0..N-1 "
            "using colormap.txt. Supports single file or batch processing in a directory."
        )
    )
    parser.add_argument(
        "--label_npy",
        type=str,
        required=False,
        help="Path to a single label npy file (H,W) or (H,W,3)",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=False,
        help=(
            "Directory containing multiple ParcelIDs_XXXXX.npy label files to batch convert. "
            "Outputs will be named ParcelIDs_XXXXX_labels.npz by default."
        ),
    )
    parser.add_argument(
        "--colormap_txt",
        type=str,
        required=True,
        help="Path to colormap.txt defining colors for each class (line index = class ID)",
    )
    parser.add_argument(
        "--label_names_json",
        type=str,
        required=False,
        help="Optional label_names.json (stored as metadata only in output npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help=(
            "Output file path for single-file mode. If ends with .npz, stores 'labels' and "
            "optional 'label_names'. If ends with .npy, stores labels array only."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help=(
            "When using --label_dir, directory to write per-file *_labels.npz. "
            "Defaults to the same directory as --label_dir."
        ),
    )

    args = parser.parse_args()

    if (args.label_npy is None) == (args.label_dir is None):
        raise ValueError("You must specify exactly one of --label_npy or --label_dir.")

    colormap_path = Path(args.colormap_txt)
    if not colormap_path.exists():
        raise FileNotFoundError(f"Colormap file not found: {colormap_path}")

    # Load label names metadata once (if provided) and reuse for all outputs.
    label_names_meta = None
    if args.label_names_json:
        ln_path = Path(args.label_names_json)
        if ln_path.exists():
            with ln_path.open("r", encoding="utf-8") as f:
                names_dict = json.load(f)
            try:
                names_list = [names_dict[str(i)] for i in range(len(names_dict))]
            except Exception:
                names_list = list(names_dict.values())
            label_names_meta = np.asarray(names_list, dtype=object)

    # Single-file mode (backward compatible with original usage)
    if args.label_npy is not None:
        if not args.output:
            raise ValueError("In single-file mode you must provide --output.")

        label_path = Path(args.label_npy)
        out_path = Path(args.output)
        convert_single_label_file(label_path, colormap_path, label_names_meta, out_path)
        return

    # Batch mode: process all ParcelIDs_*.npy in a directory
    label_dir = Path(args.label_dir)
    if not label_dir.is_dir():
        raise NotADirectoryError(f"label_dir is not a directory: {label_dir}")

    out_dir = Path(args.output_dir) if args.output_dir else label_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(label_dir.glob("ParcelIDs_*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No ParcelIDs_*.npy files found in {label_dir}")

    print(f"Found {len(npy_files)} label files in {label_dir}, writing outputs to {out_dir}")

    for npy_path in npy_files:
        stem = npy_path.stem  # e.g. ParcelIDs_00001
        out_path = out_dir / f"{stem}_labels.npz"
        print("\n==============================")
        print(f"Processing {npy_path} -> {out_path}")
        convert_single_label_file(npy_path, colormap_path, label_names_meta, out_path)


if __name__ == "__main__":
    main()
