import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.data_gee_multisource import create_gee_multisource_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on GEE multi-source dataset and export 64D embeddings",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing GEE sample_*.npz files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pt) from outputs_gee_multisource",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_gee_multisource/embeddings",
        help="Directory to save per-tile embedding npz files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (1 recommended to keep tile ordering)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Patch size used during training (H, W)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["tiny", "small", "base"],
        help=(
            "Model size of AlphaEarth used during training. "
            "Must match the model_size used in run_train_gee_multisource "
            "(e.g. 'tiny' for low-memory CPU runs)."
        ),
    )
    parser.add_argument(
        "--summary_strategy",
        type=str,
        default="full_period",
        choices=["full_period", "per_timestamp"],
        help=(
            "How to summarize over time: "
            "'full_period' (single embedding per tile over whole valid_period) or "
            "'per_timestamp' (one embedding per time step for later time-series annotation alignment)"
        ),
    )
    parser.add_argument(
        "--max_time_steps",
        type=int,
        default=None,
        help=(
            "Optional cap on number of timestamps per sample when using "
            "'per_timestamp' summary_strategy. If set, only the first N time steps are used."
        ),
    )
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    landsat_channels: int,
    sentinel1_channels: int,
    sentinel2_channels: int,
    device: torch.device,
    model_size: str,
) -> AlphaEarthFoundations:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("model_config", {})
    input_sources = model_config.get(
        "input_sources",
        {
            "landsat": landsat_channels,
            "sentinel1": sentinel1_channels,
            "sentinel2": sentinel2_channels,
        },
    )
    decode_sources = model_config.get(
        "decode_sources",
        {
            "landsat": landsat_channels,
            "sentinel1": sentinel1_channels,
            "sentinel2": sentinel2_channels,
        },
    )

    model = AlphaEarthFoundations(
        model_size=model_size,
        input_sources=input_sources,
        decode_sources=decode_sources,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataloader = create_gee_multisource_dataloader(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        normalize=True,
        shuffle=False,
    )

    # Infer channel counts from first sample
    first_sample = dataloader.dataset[0]
    landsat_channels = first_sample["source_data"]["landsat"].shape[-1]
    sentinel1_channels = first_sample["source_data"]["sentinel1"].shape[-1]
    sentinel2_channels = first_sample["source_data"]["sentinel2"].shape[-1]

    model = load_model_from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        landsat_channels=landsat_channels,
        sentinel1_channels=sentinel1_channels,
        sentinel2_channels=sentinel2_channels,
        device=device,
        model_size=args.model_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    num_samples = len(dataset)

    print(f"Running inference on {num_samples} samples from {data_dir}...")

    if args.summary_strategy == "full_period":
        # Original behaviour: one embedding per tile over its full valid_period
        global_idx = 0
        for batch_idx, batch in enumerate(dataloader):
            source_data: Dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in batch["source_data"].items()
            }
            timestamps: Dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in batch["timestamps"].items()
            }
            valid_periods = batch["valid_periods"]

            out: Dict[str, Any] = model(
                source_data=source_data,
                timestamps=timestamps,
                valid_periods=valid_periods,
            )

            embeddings = out["embeddings"].detach().cpu().numpy()  # (B, H', W', 64)
            image_embeddings = out["image_embeddings"].detach().cpu().numpy()  # (B, 64)

            batch_size = embeddings.shape[0]
            for i in range(batch_size):
                sample_idx = global_idx
                global_idx += 1

                # Map back to file name in dataset
                tile_path = dataset.files[sample_idx]

                emb = embeddings[i]  # (H', W', 64)
                img_emb = image_embeddings[i]  # (64,)

                # We also save the grid timestamps used for this sample (landsat key)
                ts = batch["timestamps"]["landsat"][i].detach().cpu().numpy()

                out_path = output_dir / f"embedding_{sample_idx:04d}.npz"
                np.savez(
                    out_path,
                    embeddings=emb,
                    image_embedding=img_emb,
                    timestamps=ts,
                    tile_file=str(tile_path),
                )

                print(
                    f"Saved embeddings to {out_path} "
                    f"(tile: {tile_path.name}, emb shape {emb.shape})"
                )
    else:
        # per_timestamp: for each tile, compute an embedding for each time step.
        # This is useful when you later have time-series annotations and
        # want to align each annotation timestamp to a corresponding 64D embedding.
        if args.batch_size != 1:
            raise ValueError(
                "summary_strategy='per_timestamp' currently requires batch_size=1 "
                "so that tiles can be processed independently."
            )

        global_idx = 0
        for batch_idx, batch in enumerate(dataloader):
            source_data: Dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in batch["source_data"].items()
            }
            timestamps: Dict[str, torch.Tensor] = {
                k: v.to(device) for k, v in batch["timestamps"].items()
            }

            # We use the Landsat grid timestamps as the reference time axis.
            ts_landsat = timestamps["landsat"][0]  # (T,)
            T = ts_landsat.shape[0]
            if args.max_time_steps is not None:
                T_eff = min(T, args.max_time_steps)
            else:
                T_eff = T

            per_ts_embeddings = []
            per_ts_img_embeddings = []

            for t_idx in range(T_eff):
                t_val = float(ts_landsat[t_idx].item())
                valid_periods = [(t_val, t_val)]  # single-sample batch

                out: Dict[str, Any] = model(
                    source_data=source_data,
                    timestamps=timestamps,
                    valid_periods=valid_periods,
                )

                emb_t = out["embeddings"][0].detach().cpu().numpy()  # (H', W', 64)
                img_emb_t = (
                    out["image_embeddings"][0].detach().cpu().numpy()
                )  # (64,)

                per_ts_embeddings.append(emb_t)
                per_ts_img_embeddings.append(img_emb_t)

            per_ts_embeddings_arr = np.stack(per_ts_embeddings, axis=0)
            per_ts_img_arr = np.stack(per_ts_img_embeddings, axis=0)
            ts_arr = ts_landsat[:T_eff].detach().cpu().numpy()

            sample_idx = global_idx
            global_idx += 1
            tile_path = dataset.files[sample_idx]

            out_path = output_dir / f"embedding_timeseries_{sample_idx:04d}.npz"
            np.savez(
                out_path,
                embeddings_per_time=per_ts_embeddings_arr,
                image_embeddings_per_time=per_ts_img_arr,
                timestamps=ts_arr,
                tile_file=str(tile_path),
            )

            print(
                f"Saved time-series embeddings to {out_path} "
                f"(tile: {tile_path.name}, T={T_eff}, "
                f"emb shape {per_ts_embeddings_arr.shape[1:]})"
            )

    print("Inference finished.")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
