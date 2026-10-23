"""Pix2Pix models and training utilities for thermal-to-RGB translation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _load_torch() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise ImportError(
            "PyTorch is required for Pix2Pix training. "
            "Install a compatible torch build, then try again."
        ) from error
    return torch, torch.nn


def create_pix2pix_models(
    input_channels: int = 1,
    output_channels: int = 3,
    base_channels: int = 64,
) -> tuple[Any, Any]:
    """Create a U-Net generator and a PatchGAN discriminator."""
    torch, nn = _load_torch()
    if min(input_channels, output_channels, base_channels) <= 0:
        raise ValueError("Model channel counts must be positive.")

    class DownBlock(nn.Module):
        def __init__(self, channels_in: int, channels_out: int, normalize: bool = True):
            super().__init__()
            layers: list[Any] = [
                nn.Conv2d(channels_in, channels_out, 4, 2, 1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(channels_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.layers = nn.Sequential(*layers)

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class UpBlock(nn.Module):
        def __init__(self, channels_in: int, channels_out: int, dropout: bool = False):
            super().__init__()
            layers: list[Any] = [
                nn.ConvTranspose2d(channels_in, channels_out, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(channels_out),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            self.layers = nn.Sequential(*layers)

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class UNetGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = DownBlock(input_channels, base_channels, normalize=False)
            self.down2 = DownBlock(base_channels, base_channels * 2)
            self.down3 = DownBlock(base_channels * 2, base_channels * 4)
            self.down4 = DownBlock(base_channels * 4, base_channels * 8)
            self.down5 = DownBlock(base_channels * 8, base_channels * 8)
            self.down6 = DownBlock(base_channels * 8, base_channels * 8)
            self.down7 = DownBlock(base_channels * 8, base_channels * 8)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(base_channels * 8, base_channels * 8, 4, 2, 1),
                nn.ReLU(inplace=True),
            )
            self.up1 = UpBlock(base_channels * 8, base_channels * 8, True)
            self.up2 = UpBlock(base_channels * 16, base_channels * 8, True)
            self.up3 = UpBlock(base_channels * 16, base_channels * 8, True)
            self.up4 = UpBlock(base_channels * 16, base_channels * 8)
            self.up5 = UpBlock(base_channels * 16, base_channels * 4)
            self.up6 = UpBlock(base_channels * 8, base_channels * 2)
            self.up7 = UpBlock(base_channels * 4, base_channels)
            self.output = nn.Sequential(
                nn.ConvTranspose2d(base_channels * 2, output_channels, 4, 2, 1),
                nn.Tanh(),
            )

        def forward(self, inputs: Any) -> Any:
            skips = [
                self.down1(inputs),
            ]
            skips.append(self.down2(skips[-1]))
            skips.append(self.down3(skips[-1]))
            skips.append(self.down4(skips[-1]))
            skips.append(self.down5(skips[-1]))
            skips.append(self.down6(skips[-1]))
            skips.append(self.down7(skips[-1]))
            features = self.bottleneck(skips[-1])
            features = self.up1(features)
            features = self.up2(torch.cat((features, skips[-1]), dim=1))
            features = self.up3(torch.cat((features, skips[-2]), dim=1))
            features = self.up4(torch.cat((features, skips[-3]), dim=1))
            features = self.up5(torch.cat((features, skips[-4]), dim=1))
            features = self.up6(torch.cat((features, skips[-5]), dim=1))
            features = self.up7(torch.cat((features, skips[-6]), dim=1))
            return self.output(torch.cat((features, skips[-7]), dim=1))

    class PatchGANDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = input_channels + output_channels
            self.network = nn.Sequential(
                nn.Conv2d(channels, base_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1, bias=False),
                nn.InstanceNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels * 8, 1, 4, 1, 1),
            )

        def forward(self, thermal: Any, rgb: Any) -> Any:
            return self.network(torch.cat((thermal, rgb), dim=1))

    return UNetGenerator(), PatchGANDiscriminator()


def create_pix2pix_dataloader(
    dataset_root: Path,
    split: str,
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> Any:
    """Load visible RGB plus thermal inputs and RGB targets in [-1, 1]."""
    torch, _ = _load_torch()
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test.")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative.")

    class PairedPatchDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.thermal_files = sorted(
                (dataset_root / split / "input_thermal").glob("*.png")
            )
            self.visible_directory = dataset_root / split / "input_rgb"
            self.rgb_directory = dataset_root / split / "rgb"
            if not self.thermal_files:
                raise FileNotFoundError(
                    f"No thermal patches found in "
                    f"{dataset_root / split / 'input_thermal'}"
                )
            missing = [
                path.name for path in self.thermal_files
                if not (self.visible_directory / path.name).is_file()
                or not (self.rgb_directory / path.name).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"Missing RGB patch, including: {missing[0]}")

        def __len__(self) -> int:
            return len(self.thermal_files)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            thermal_path = self.thermal_files[index]
            visible_path = self.visible_directory / thermal_path.name
            rgb_path = self.rgb_directory / thermal_path.name
            thermal_image = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
            visible_image = cv2.imread(str(visible_path), cv2.IMREAD_COLOR)
            rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if thermal_image is None or visible_image is None or rgb_image is None:
                raise OSError(f"Unable to read paired patch: {thermal_path.name}")
            visible_image = cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            thermal_tensor = torch.from_numpy(thermal_image.copy()).float().unsqueeze(0)
            visible_tensor = torch.from_numpy(visible_image.copy()).float().permute(2, 0, 1)
            rgb_tensor = torch.from_numpy(rgb_image.copy()).float().permute(2, 0, 1)
            fusion_tensor = torch.cat((visible_tensor, thermal_tensor), dim=0)
            return fusion_tensor / 127.5 - 1.0, rgb_tensor / 127.5 - 1.0

    if shuffle is None:
        shuffle = split == "train"
    return torch.utils.data.DataLoader(
        PairedPatchDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _select_device(torch: Any, device: str | None) -> Any:
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device(selected)


def train_pix2pix(
    dataset_root: Path,
    checkpoint_directory: Path,
    epochs: int = 50,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    lambda_l1: float = 100.0,
    device: str | None = None,
    num_workers: int = 0,
    base_channels: int = 64,
    mixed_precision: bool = True,
    resume_checkpoint: Path | None = None,
) -> Path:
    """Train Pix2Pix and save the latest and best checkpoints."""
    torch, _ = _load_torch()
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0 or lambda_l1 <= 0:
        raise ValueError("epochs, batch_size, learning_rate, and lambda_l1 must be positive.")

    selected_device = _select_device(torch, device)
    use_mixed_precision = mixed_precision and selected_device.type == "cuda"
    generator, discriminator = create_pix2pix_models(
        input_channels=4,
        output_channels=3,
        base_channels=base_channels,
    )
    generator.to(selected_device)
    discriminator.to(selected_device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    start_epoch = 1
    best_validation = float("inf")
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=selected_device)
        checkpoint_configuration = checkpoint.get("configuration", {})
        if int(checkpoint_configuration.get("input_channels", 1)) != 4:
            raise ValueError(
                "The resume checkpoint is for the previous one-channel Pix2Pix "
                "model. Start four-channel fusion training without resuming it."
            )
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint.get("best_validation", best_validation))

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    train_loader = create_pix2pix_dataloader(
        dataset_root, "train", batch_size, True, num_workers
    )
    validation_loader = create_pix2pix_dataloader(
        dataset_root, "validation", batch_size, False, num_workers
    )
    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    reconstruction_loss = torch.nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    history: list[dict[str, float | int]] = []

    for epoch in range(start_epoch, epochs + 1):
        generator.train()
        discriminator.train()
        generator_total = 0.0
        discriminator_total = 0.0
        for thermal, real_rgb in train_loader:
            thermal = thermal.to(selected_device, non_blocking=True)
            real_rgb = real_rgb.to(selected_device, non_blocking=True)

            discriminator_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type, enabled=use_mixed_precision
            ):
                fake_rgb = generator(thermal)
                real_prediction = discriminator(thermal, real_rgb)
                fake_prediction = discriminator(thermal, fake_rgb.detach())
                discriminator_loss = (
                    adversarial_loss(real_prediction, torch.ones_like(real_prediction))
                    + adversarial_loss(fake_prediction, torch.zeros_like(fake_prediction))
                ) * 0.5
            scaler.scale(discriminator_loss).backward()
            scaler.step(discriminator_optimizer)

            generator_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type, enabled=use_mixed_precision
            ):
                fake_prediction = discriminator(thermal, fake_rgb)
                generator_loss = adversarial_loss(
                    fake_prediction, torch.ones_like(fake_prediction)
                )
                generator_loss = generator_loss + lambda_l1 * reconstruction_loss(
                    fake_rgb, real_rgb
                )
            scaler.scale(generator_loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()
            generator_total += float(generator_loss.detach().item())
            discriminator_total += float(discriminator_loss.detach().item())

        validation_loss = _validation_l1(
            torch, generator, validation_loader, selected_device
        )
        epoch_record = {
            "epoch": epoch,
            "generator_loss": generator_total / len(train_loader),
            "discriminator_loss": discriminator_total / len(train_loader),
            "validation_l1": validation_loss,
        }
        history.append(epoch_record)
        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "best_validation": min(best_validation, validation_loss),
            "configuration": {
                "input_channels": 4,
                "output_channels": 3,
                "base_channels": base_channels,
                "lambda_l1": lambda_l1,
            },
        }
        latest_path = checkpoint_directory / "pix2pix_latest.pt"
        torch.save(checkpoint, latest_path)
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(checkpoint, checkpoint_directory / "pix2pix_best.pt")
        (checkpoint_directory / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"Epoch {epoch}/{epochs} | "
            f"G: {epoch_record['generator_loss']:.4f} | "
            f"D: {epoch_record['discriminator_loss']:.4f} | "
            f"Val L1: {validation_loss:.4f}"
        )

    return checkpoint_directory / "pix2pix_best.pt"


def _validation_l1(torch: Any, generator: Any, loader: Any, device: Any) -> float:
    generator.eval()
    total = 0.0
    with torch.inference_mode():
        for thermal, real_rgb in loader:
            fake_rgb = generator(thermal.to(device, non_blocking=True))
            total += float(torch.nn.functional.l1_loss(
                fake_rgb, real_rgb.to(device, non_blocking=True)
            ).item())
    return total / len(loader)
