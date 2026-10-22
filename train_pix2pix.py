"""Train the Pix2Pix thermal-to-RGB translation model for one scene."""

import os
from pathlib import Path

from pix2pix_model import train_pix2pix
DATASET_ROOT = Path(__file__).resolve().parent / "Pix2Pix_Combined_Dataset"
CHECKPOINT_ROOT = Path(__file__).resolve().parent / "Pix2Pix_Combined_Dataset" / "checkpoints"


def main() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Combined Pix2Pix dataset was not found: {DATASET_ROOT}\n"
            "Run prepare_combined_pix2pix_dataset.py first."
        )
    resume_value = os.environ.get("PIX2PIX_RESUME_CHECKPOINT")
    resume_checkpoint = Path(resume_value).expanduser() if resume_value else None
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(
            f"Pix2Pix resume checkpoint was not found: {resume_checkpoint}"
        )
    checkpoint_path = train_pix2pix(
        dataset_root=DATASET_ROOT,
        checkpoint_directory=CHECKPOINT_ROOT,
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
