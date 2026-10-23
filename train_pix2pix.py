"""Train the shared four-channel Pix2Pix fusion model."""

import os
from pathlib import Path

from pix2pix_model import train_pix2pix


PROJECT_DATASET_ROOT = Path(__file__).resolve().parent / "Pix2Pix_Combined_Dataset"
DEFAULT_DATASET_ROOT = Path(r"D:\datasets\Pix2Pix_Combined_Dataset")


def find_pix2pix_dataset() -> Path:
    """Find the shared dataset using an override, data-drive, or local path."""
    configured_path = os.environ.get("PIX2PIX_DATASET_ROOT")
    candidates = (
        [Path(configured_path).expanduser()]
        if configured_path
        else [DEFAULT_DATASET_ROOT, PROJECT_DATASET_ROOT]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Combined Pix2Pix dataset was not found. Searched:\n"
        f"{searched}\n"
        "Set PIX2PIX_DATASET_ROOT to the dataset folder."
    )


def main() -> None:
    dataset_root = find_pix2pix_dataset()
    checkpoint_root = dataset_root / "checkpoints"
    resume_value = os.environ.get("PIX2PIX_RESUME_CHECKPOINT")
    resume_checkpoint = Path(resume_value).expanduser() if resume_value else None
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(
            f"Pix2Pix resume checkpoint was not found: {resume_checkpoint}"
        )
    checkpoint_path = train_pix2pix(
        dataset_root=dataset_root,
        checkpoint_directory=checkpoint_root,
        epochs=50,
        batch_size=1,
        learning_rate=2e-4,
        lambda_l1=100.0,
        device="cuda",
        base_channels=64,
        mixed_precision=True,
        resume_checkpoint=resume_checkpoint,
    )
    print(f"Best Pix2Pix checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
