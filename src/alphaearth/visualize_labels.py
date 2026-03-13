import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alphaearth.convert_label_to_indices import load_colormap


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize integer label mask using colormap.txt and save as JPG/PNG."
        )
    )
    parser.add_argument(
        "--labels_file",
        type=str,
        required=True,
        help=(
            "Path to labels file. If .npz, expects key 'labels'. "
            "If .npy, uses the array directly."
        ),
    )
    parser.add_argument(
        "--colormap_txt",
        type=str,
        required=True,
        help="Path to colormap.txt defining (r,g,b) per class (0..N-1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output image path (e.g., .jpg or .png)",
    )

    args = parser.parse_args()

    labels_path = Path(args.labels_file)
    cmap_path = Path(args.colormap_txt)
    out_path = Path(args.output)

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not cmap_path.exists():
        raise FileNotFoundError(f"Colormap file not found: {cmap_path}")

    print(f"Loading labels from {labels_path}...")
    if labels_path.suffix == ".npz":
        data = np.load(labels_path, allow_pickle=True)
        if "labels" not in data:
            raise KeyError(f"'labels' key not found in {labels_path}")
        labels = data["labels"]
    else:
        labels = np.load(labels_path)

    if labels.ndim != 2:
        raise ValueError(f"Expected labels with shape (H,W), got {labels.shape}")

    labels = labels.astype(np.int64)
    print(f"  labels shape: {labels.shape}, unique IDs: {np.unique(labels)}")

    print(f"Loading colormap from {cmap_path}...")
    colormap = load_colormap(cmap_path)  # (N,3) in [0,1]
    print(f"  colormap has {colormap.shape[0]} classes")

    # Clip labels to valid range just in case
    max_class = colormap.shape[0] - 1
    labels_clipped = np.clip(labels, 0, max_class)

    # Map each class ID to its RGB color
    rgb = colormap[labels_clipped]  # (H,W,3), float [0,1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving label visualization to {out_path}...")
    plt.imsave(out_path, rgb)
    print("Done.")


if __name__ == "__main__":
    main()
