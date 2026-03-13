import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


def load_colormap(colormap_path: Path) -> np.ndarray:
    """Load colormap.txt -> (N,3) float32 in [0,1]."""
    rows: list[tuple[float, float, float]] = []
    with colormap_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Expect lines like: (r, g, b),
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


def color_mask_to_labels(
    color_mask: np.ndarray,
    colormap: np.ndarray,
) -> np.ndarray:
    """Map (H,W,3) float color mask to (H,W) int class indices via nearest color.

    Args:
        color_mask: (H,W,3) in [0,1] or [0,255]
        colormap: (N,3) in [0,1]
    """
    if color_mask.ndim != 3 or color_mask.shape[-1] != 3:
        raise ValueError(f"Expected color_mask with shape (H,W,3), got {color_mask.shape}")

    cm = colormap.astype(np.float32)
    img = color_mask.astype(np.float32)

    # Normalize image to [0,1] if needed
    max_val = img.max()
    if max_val > 1.0 + 1e-3:
        img = img / 255.0

    H, W, _ = img.shape
    img_flat = img.reshape(-1, 3)  # (Npix,3)

    # Compute squared distances to each colormap row
    # diff: (Npix, Ncolors, 3)
    diff = img_flat[:, None, :] - cm[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)  # (Npix, Ncolors)
    idx = np.argmin(dist2, axis=1).astype(np.int64)  # (Npix,)

    return idx.reshape(H, W)


def prepare_features_from_embeddings(emb_npz: Path) -> np.ndarray:
    """Load embeddings npz and return features as (C,H,W) float32.

    Supports two formats:
      - embeddings: (H,W,64)
      - embeddings_per_time: (T,H,W,64) -> (T*64,H,W)
    """
    data = np.load(emb_npz)

    if "embeddings_per_time" in data:
        e = data["embeddings_per_time"]  # (T,H,W,64)
        if e.ndim != 4 or e.shape[-1] != 64:
            raise ValueError(f"Unexpected embeddings_per_time shape {e.shape} in {emb_npz}")
        # (T,H,W,64) -> (T,64,H,W) -> (T*64,H,W)
        e_chw = np.transpose(e, (0, 3, 1, 2))
        T, C, H, W = e_chw.shape
        feats = e_chw.reshape(T * C, H, W)
    elif "embeddings" in data:
        e = data["embeddings"]  # (H,W,64)
        if e.ndim != 3 or e.shape[-1] != 64:
            raise ValueError(f"Unexpected embeddings shape {e.shape} in {emb_npz}")
        # (H,W,64) -> (64,H,W)
        feats = np.transpose(e, (2, 0, 1))
    else:
        raise ValueError(
            f"Neither 'embeddings_per_time' nor 'embeddings' found in {emb_npz}"
        )

    return feats.astype(np.float32)


def resize_labels_to(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Resize label mask (H_l,W_l) to match features (C,H_f,W_f) using nearest neighbor."""
    C, H_f, W_f = features.shape
    H_l, W_l = labels.shape

    if (H_f, W_f) == (H_l, W_l):
        return labels

    zoom_h = H_f / H_l
    zoom_w = W_f / W_l
    resized = zoom(labels, zoom=(zoom_h, zoom_w), order=0)
    resized = resized.astype(labels.dtype)
    return resized


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a segmentation dataset from 64D AEF embeddings and "
            "color-coded label mask. Outputs a single npz with features "
            "(C,H,W) and labels (H,W) suitable for U-Net training."
        )
    )
    parser.add_argument(
        "--embeddings_npz",
        type=str,
        required=True,
        help=(
            "Path to an embeddings npz file (either embedding_XXXX.npz or "
            "embedding_timeseries_XXXX.npz)."
        ),
    )
    parser.add_argument(
        "--label_npy",
        type=str,
        required=True,
        help=(
            "Path to a color label npy file (H,W,3), whose colors follow "
            "colormap.txt."
        ),
    )
    parser.add_argument(
        "--colormap_txt",
        type=str,
        required=True,
        help="Path to colormap.txt (each line is an (r,g,b) tuple).",
    )
    parser.add_argument(
        "--label_names_json",
        type=str,
        required=False,
        help="Optional path to label_names.json; stored as metadata only.",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        required=True,
        help=(
            "Output npz path. Will contain 'features' (C,H,W), 'labels' (H,W), "
            "and optional 'label_names' array of strings."
        ),
    )

    args = parser.parse_args()

    emb_path = Path(args.embeddings_npz)
    label_path = Path(args.label_npy)
    colormap_path = Path(args.colormap_txt)
    out_path = Path(args.output_npz)

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings npz not found: {emb_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label npy not found: {label_path}")
    if not colormap_path.exists():
        raise FileNotFoundError(f"Colormap file not found: {colormap_path}")

    print(f"Loading embeddings from {emb_path}...")
    features = prepare_features_from_embeddings(emb_path)  # (C,H_e,W_e)
    C, H_e, W_e = features.shape
    print(f"  features shape: C={C}, H={H_e}, W={W_e}")

    print(f"Loading color labels from {label_path}...")
    color_labels = np.load(label_path)
    if color_labels.ndim != 3 or color_labels.shape[-1] != 3:
        raise ValueError(
            f"Expected label_npy with shape (H,W,3), got {color_labels.shape}"
        )

    print(f"Loading colormap from {colormap_path}...")
    colormap = load_colormap(colormap_path)
    print(f"  colormap has {colormap.shape[0]} classes")

    print("Mapping colors to class indices (0..N-1)...")
    labels_hw = color_mask_to_labels(color_labels, colormap)  # (H_l,W_l)

    print("Resizing labels to match embedding spatial resolution (if needed)...")
    labels_resized = resize_labels_to(features, labels_hw)
    H_l2, W_l2 = labels_resized.shape
    print(f"  resized labels shape: H={H_l2}, W={W_l2}")

    meta: dict[str, object] = {}
    if args.label_names_json:
        ln_path = Path(args.label_names_json)
        if ln_path.exists():
            with ln_path.open("r", encoding="utf-8") as f:
                names_dict = json.load(f)
            # Store as ordered list by integer key if possible
            try:
                names_list = [names_dict[str(i)] for i in range(len(names_dict))]
            except Exception:
                names_list = list(names_dict.values())
            meta["label_names"] = np.asarray(names_list, dtype=object)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving segmentation dataset to {out_path}...")
    if meta:
        np.savez(out_path, features=features, labels=labels_resized.astype(np.int64), **meta)
    else:
        np.savez(out_path, features=features, labels=labels_resized.astype(np.int64))

    print("Done. You can now train a U-Net with C input channels = features.shape[0].")


if __name__ == "__main__":
    main()
