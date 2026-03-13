import argparse
from pathlib import Path

import torch

from alphaearth.architecture.aef_module import AlphaEarthFoundations
from alphaearth.training import create_trainer
from alphaearth.data_gee_multisource import create_gee_multisource_dataloader


def parse_reconstruction_sources(value: str) -> list[str]:
    sources = [item.strip() for item in value.split(",") if item.strip()]
    if not sources:
        raise argparse.ArgumentTypeError("At least one reconstruction source is required")

    allowed = {"landsat", "sentinel1", "sentinel2"}
    invalid = [item for item in sources if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported reconstruction sources: {invalid}. Allowed values: {sorted(allowed)}"
        )
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AlphaEarth on GEE L8/S1/S2 multi-source dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/gee_multisource",
        help="Directory containing GEE .npz samples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Spatial patch size (H, W)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides --epochs if set)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps for learning rate",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=20,
        help="Log every N steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_gee_multisource",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not provided",
    )
    parser.add_argument(
        "--reconstruction_weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss term (default 1.0)",
    )
    parser.add_argument(
        "--uniformity_weight",
        type=float,
        default=0.01,
        help="Weight for uniformity loss term (default 0.01)",
    )
    parser.add_argument(
        "--consistency_weight",
        type=float,
        default=0.005,
        help="Weight for teacher-student consistency loss term (default 0.005)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["tiny", "small", "base"],
        help=(
            "Model size for AlphaEarth encoder/decoder. "
            "Use 'tiny' on CPU or low-memory machines to reduce parameter "
            "count and memory usage. Default is 'small'."
        ),
    )
    parser.add_argument(
        "--reconstruction_sources",
        type=parse_reconstruction_sources,
        default=["landsat", "sentinel1", "sentinel2"],
        help=(
            "Comma-separated list of sources to reconstruct. "
            "By default all three sources are reconstructed, while Landsat/S1/S2 are always concatenated "
            "as encoder inputs before STP processing."
        ),
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Loading GEE multi-source dataset from {data_dir}")

    dataloader = create_gee_multisource_dataloader(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        normalize=True,
        shuffle=True,
    )

    dataset_size = len(dataloader.dataset)
    steps_per_epoch = dataset_size // args.batch_size
    if steps_per_epoch == 0:
        raise ValueError("Dataset too small for given batch size.")

    if args.max_steps is None:
        max_steps = args.epochs * steps_per_epoch
    else:
        max_steps = args.max_steps

    print(f"Dataset: {dataset_size} samples, {steps_per_epoch} steps/epoch")
    print(f"Training for {max_steps} steps ({max_steps / steps_per_epoch:.2f} epochs)")

    landsat_channels = dataloader.dataset[0]["source_data"]["landsat"].shape[-1]
    sentinel1_channels = dataloader.dataset[0]["source_data"]["sentinel1"].shape[-1]
    sentinel2_channels = dataloader.dataset[0]["source_data"]["sentinel2"].shape[-1]
    channel_map = {
        "landsat": landsat_channels,
        "sentinel1": sentinel1_channels,
        "sentinel2": sentinel2_channels,
    }
    decode_sources = {name: channel_map[name] for name in args.reconstruction_sources}

    model = AlphaEarthFoundations(
        model_size=args.model_size,
        input_sources={
            "landsat": landsat_channels,
            "sentinel1": sentinel1_channels,
            "sentinel2": sentinel2_channels,
        },
        decode_sources=decode_sources,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print("\n" + "=" * 80)
    print("MODEL INFORMATION")
    print("=" * 80)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Model size: {args.model_size}")
    print(f"Input sources: {model.input_sources}")
    print(f"Decode sources: {model.decode_sources}")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    param_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Model size (float32): {param_size_mb:.2f} MB")
    print("\n" + "-" * 80)
    print("MODEL ARCHITECTURE")
    print("-" * 80)
    print(model)
    print("=" * 80 + "\n")

    trainer = create_trainer(
        model=model,
        dataloader=dataloader,
        text_adapter=None,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        reconstruction_weight=args.reconstruction_weight,
        uniformity_weight=args.uniformity_weight,
        consistency_weight=args.consistency_weight,
    )

    trainer.max_steps = max_steps
    trainer.warmup_steps = args.warmup_steps

    print(f"Starting training for {max_steps} steps...")
    print(f"Output directory: {args.output_dir}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.train(max_steps=max_steps, log_every=args.log_every)

    print("Training run finished.")


if __name__ == "__main__":
    main()
