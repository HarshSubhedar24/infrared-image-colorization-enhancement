"""
==============================================================
BE FINAL YEAR PROJECT

STEP 03
ESRGAN Dataset Preparation

Creates paired high-resolution (HR) and low-resolution (LR) thermal
patches from the grayscale output of Step 02.

The implementation is window-based: the source raster is never loaded
into memory as a complete image.
==============================================================
"""

from __future__ import annotations

import gc
import importlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
from scene_selection import select_scene_directory


# ==========================================================
# CONFIGURATION
# ==========================================================

DEFAULT_BASE_PATH = Path(
    r"D:\datasets"
)
PATCH_SIZE = 512
SCALE_FACTOR = 4
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42
PREVIEW_COUNT = 5


def _load_torch() -> tuple[object, object]:
    """Load PyTorch only when the ESRGAN model is requested."""
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise ImportError(
            "PyTorch is required to create the ESRGAN model. "
            "Install a compatible torch build, then run this function again."
        ) from error
    return torch, torch.nn


def create_esrgan_models(
    scale_factor: int = SCALE_FACTOR,
    input_channels: int = 1,
    output_channels: int = 1,
    feature_channels: int = 64,
    rrdb_blocks: int = 23,
    discriminator_channels: int | None = None,
) -> tuple[object, object]:
    """Create the ESRGAN generator and discriminator.

    The generator uses Residual-in-Residual Dense Blocks (RRDBs) and
    sub-pixel convolution upsampling. Single-channel thermal images are
    supported by default.
    """
    torch, nn = _load_torch()

    if scale_factor not in (2, 4, 8):
        raise ValueError("scale_factor must be 2, 4, or 8.")
    if min(input_channels, output_channels, feature_channels, rrdb_blocks) <= 0:
        raise ValueError("Model channel counts and rrdb_blocks must be positive.")
    if discriminator_channels is None:
        discriminator_channels = feature_channels
    if discriminator_channels <= 0:
        raise ValueError("discriminator_channels must be positive.")

    class ResidualDenseBlock(nn.Module):
        def __init__(self, channels: int, growth_channels: int = 32) -> None:
            super().__init__()
            self.layers = nn.ModuleList()
            for layer_index in range(5):
                input_size = channels + layer_index * growth_channels
                output_size = growth_channels if layer_index < 4 else channels
                self.layers.append(nn.Conv2d(input_size, output_size, 3, 1, 1))

        def forward(self, inputs: object) -> object:
            features = [inputs]
            for layer_index, layer in enumerate(self.layers):
                output = layer(torch.cat(features, dim=1))
                if layer_index < 4:
                    output = torch.nn.functional.leaky_relu(
                        output,
                        negative_slope=0.2,
                        inplace=True,
                    )
                features.append(output)
            return inputs + 0.2 * features[-1]

    class RRDB(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                ResidualDenseBlock(channels),
                ResidualDenseBlock(channels),
                ResidualDenseBlock(channels),
            )

        def forward(self, inputs: object) -> object:
            return inputs + 0.2 * self.blocks(inputs)

    class ESRGANGenerator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first_convolution = nn.Conv2d(
                input_channels, feature_channels, 3, 1, 1
            )
            self.body = nn.Sequential(
                *[RRDB(feature_channels) for _ in range(rrdb_blocks)]
            )
            self.body_convolution = nn.Conv2d(
                feature_channels, feature_channels, 3, 1, 1
            )

            upsample_layers: list[object] = []
            for _ in range(scale_factor.bit_length() - 1):
                upsample_layers.extend(
                    [
                        nn.Conv2d(feature_channels, feature_channels * 4, 3, 1, 1),
                        nn.PixelShuffle(2),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
            self.upsampling = nn.Sequential(*upsample_layers)
            self.high_resolution_convolution = nn.Conv2d(
                feature_channels, feature_channels, 3, 1, 1
            )
            self.output_convolution = nn.Conv2d(
                feature_channels, output_channels, 3, 1, 1
            )

        def forward(self, inputs: object) -> object:
            first_features = self.first_convolution(inputs)
            body_features = self.body_convolution(self.body(first_features))
            features = first_features + body_features
            features = self.upsampling(features)
            features = torch.nn.functional.leaky_relu(
                self.high_resolution_convolution(features),
                negative_slope=0.2,
                inplace=True,
            )
            return self.output_convolution(features)

    class ESRGANDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[object] = []
            channels = discriminator_channels
            layers.extend(
                [
                    nn.Conv2d(input_channels, channels, 3, 1, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            for block_index in range(4):
                next_channels = min(channels * 2, 128)
                layers.extend(
                    [
                        nn.Conv2d(channels, next_channels, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )
                channels = next_channels
            layers.extend(
                [
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(channels, 1),
                ]
            )
            self.network = nn.Sequential(*layers)

        def forward(self, inputs: object) -> object:
            return self.network(inputs)

    return ESRGANGenerator(), ESRGANDiscriminator()


def count_model_parameters(model: object) -> int:
    """Return the number of trainable parameters in a PyTorch model."""
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        raise TypeError("model must provide a PyTorch parameters() method.")
    return sum(
        parameter.numel()
        for parameter in parameters()
        if getattr(parameter, "requires_grad", False)
    )


def _load_torch_training_tools() -> tuple[object, object]:
    """Load PyTorch and its data utilities for training and inference."""
    torch, nn = _load_torch()
    try:
        import torch.utils.data as data
    except ImportError as error:
        raise ImportError("PyTorch data utilities are required for ESRGAN training.") from error
    return torch, data


def create_esrgan_dataloader(
    dataset_root: Path,
    split: str,
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 0,
    hr_crop_size: int | None = None,
) -> object:
    """Create a DataLoader for paired grayscale LR and HR PNG patches."""
    torch, data = _load_torch_training_tools()
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test.")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative.")

    class ThermalPatchDataset(data.Dataset):
        def __init__(self) -> None:
            self.lr_files = sorted((dataset_root / split / "LR").glob("*.png"))
            self.hr_directory = dataset_root / split / "HR"
            if not self.lr_files:
                raise FileNotFoundError(f"No LR patches found in {dataset_root / split / 'LR'}")
            missing_hr = [
                path.name for path in self.lr_files
                if not (self.hr_directory / path.name).is_file()
            ]
            if missing_hr:
                raise FileNotFoundError(f"Missing HR patches, including: {missing_hr[0]}")

        def __len__(self) -> int:
            return len(self.lr_files)

        def __getitem__(self, index: int) -> tuple[object, object]:
            lr_image = cv2.imread(str(self.lr_files[index]), cv2.IMREAD_GRAYSCALE)
            hr_path = self.hr_directory / self.lr_files[index].name
            hr_image = cv2.imread(str(hr_path), cv2.IMREAD_GRAYSCALE)
            if lr_image is None or hr_image is None:
                raise OSError(f"Unable to read paired patch: {self.lr_files[index].name}")
            lr_tensor = torch.from_numpy(lr_image.copy()).float().unsqueeze(0) / 255.0
            hr_tensor = torch.from_numpy(hr_image.copy()).float().unsqueeze(0) / 255.0
            if hr_crop_size is not None:
                if hr_crop_size <= 0 or hr_crop_size > hr_tensor.shape[-1]:
                    raise ValueError(
                        "hr_crop_size must be positive and no larger than the HR patch."
                    )
                lr_crop_size = hr_crop_size // SCALE_FACTOR
                if lr_crop_size <= 0:
                    raise ValueError("hr_crop_size must be at least the scale factor.")
                max_lr_top = lr_tensor.shape[-2] - lr_crop_size
                max_lr_left = lr_tensor.shape[-1] - lr_crop_size
                lr_top = random.randint(0, max_lr_top)
                lr_left = random.randint(0, max_lr_left)
                top = lr_top * SCALE_FACTOR
                left = lr_left * SCALE_FACTOR
                hr_tensor = hr_tensor[
                    :, top:top + hr_crop_size, left:left + hr_crop_size
                ]
                lr_tensor = lr_tensor[
                    :, lr_top:lr_top + lr_crop_size, lr_left:lr_left + lr_crop_size
                ]
            return lr_tensor, hr_tensor

    if shuffle is None:
        shuffle = split == "train"
    return data.DataLoader(
        ThermalPatchDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _select_device(torch: object, device: str | None) -> object:
    """Resolve a requested device and fail clearly for unavailable CUDA."""
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device(selected)


def train_esrgan(
    dataset_root: Path,
    epochs: int = 50,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    checkpoint_directory: Path | None = None,
    resume_checkpoint: Path | None = None,
    device: str | None = None,
    num_workers: int = 0,
    feature_channels: int = 8,
    rrdb_blocks: int = 1,
    discriminator_channels: int = 8,
    mixed_precision: bool = True,
    train_hr_crop_size: int = 64,
    optimizer_name: str = "sgd",
    adversarial_training: bool = False,
) -> Path:
    """Train the generator and save a resumable checkpoint after every epoch.

    On GPUs with limited VRAM, the default content-only mode avoids retaining
    the discriminator graph during generator backpropagation. Set
    ``adversarial_training=True`` only when sufficient VRAM is available.
    """
    torch, _ = _load_torch_training_tools()
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive.")
    if train_hr_crop_size <= 0 or train_hr_crop_size % SCALE_FACTOR != 0:
        raise ValueError("train_hr_crop_size must be positive and divisible by SCALE_FACTOR.")
    if optimizer_name not in ("sgd", "adam"):
        raise ValueError("optimizer_name must be 'sgd' or 'adam'.")

    selected_device = _select_device(torch, device)
    use_mixed_precision = mixed_precision and selected_device.type == "cuda"
    generator, discriminator = create_esrgan_models(
        scale_factor=SCALE_FACTOR,
        feature_channels=feature_channels,
        rrdb_blocks=rrdb_blocks,
        discriminator_channels=discriminator_channels,
    )
    generator.to(selected_device)
    discriminator.to(selected_device)

    if optimizer_name == "sgd":
        optimizer_factory = lambda parameters: torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            foreach=False,
        )
    else:
        optimizer_factory = lambda parameters: torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.99),
            foreach=False,
            fused=False,
        )
    generator_optimizer = optimizer_factory(generator.parameters())
    discriminator_optimizer = optimizer_factory(discriminator.parameters())
    start_epoch = 1
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=selected_device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1

    output_directory = checkpoint_directory or (dataset_root / "checkpoints")
    output_directory.mkdir(parents=True, exist_ok=True)
    train_loader = create_esrgan_dataloader(
        dataset_root, "train", batch_size, True, num_workers, train_hr_crop_size
    )
    validation_loader = create_esrgan_dataloader(
        dataset_root, "validation", batch_size, False, num_workers, train_hr_crop_size
    )
    reconstruction_loss = torch.nn.L1Loss()
    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    for epoch in range(start_epoch, start_epoch + epochs):
        generator.train()
        discriminator.train()
        generator_loss_total = 0.0
        discriminator_loss_total = 0.0

        for lr_images, hr_images in train_loader:
            lr_images = lr_images.to(selected_device, non_blocking=True)
            hr_images = hr_images.to(selected_device, non_blocking=True)

            generator_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=selected_device.type,
                enabled=use_mixed_precision,
            ):
                fake_images = generator(lr_images)
                content_loss = reconstruction_loss(fake_images, hr_images)
            generator_loss = content_loss
            if adversarial_training:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
                # Keep the discriminator in float32 for Windows CUDA/cuBLAS.
                fake_predictions_for_generator = discriminator(fake_images.float())
                generator_adversarial_loss = adversarial_loss(
                    fake_predictions_for_generator,
                    torch.ones_like(fake_predictions_for_generator),
                )
                generator_loss = content_loss + 1e-3 * generator_adversarial_loss
            scaler.scale(generator_loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            discriminator_loss = torch.zeros((), device=selected_device)
            if adversarial_training:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)
                discriminator_optimizer.zero_grad(set_to_none=True)
                real_predictions = discriminator(hr_images.float())
                fake_predictions = discriminator(fake_images.detach().float())
                discriminator_loss = 0.5 * (
                    adversarial_loss(real_predictions, torch.ones_like(real_predictions))
                    + adversarial_loss(fake_predictions, torch.zeros_like(fake_predictions))
                )
                scaler.scale(discriminator_loss).backward()
                scaler.step(discriminator_optimizer)
                scaler.update()

            generator_loss_total += float(generator_loss.detach().cpu())
            discriminator_loss_total += float(discriminator_loss.detach().cpu())

        validation_loss = evaluate_esrgan(generator, validation_loader, selected_device)
        checkpoint_path = output_directory / f"esrgan_epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "generator_optimizer": generator_optimizer.state_dict(),
                "discriminator_optimizer": discriminator_optimizer.state_dict(),
                "scale_factor": SCALE_FACTOR,
                "feature_channels": feature_channels,
                "rrdb_blocks": rrdb_blocks,
                "discriminator_channels": discriminator_channels,
                "optimizer_name": optimizer_name,
                "adversarial_training": adversarial_training,
                "train_generator_loss": generator_loss_total / max(1, len(train_loader)),
                "train_discriminator_loss": discriminator_loss_total / max(1, len(train_loader)),
                "validation_l1_loss": validation_loss,
            },
            checkpoint_path,
        )
        print(
            f"Epoch {epoch}: generator={generator_loss_total / max(1, len(train_loader)):.6f}, "
            f"discriminator={discriminator_loss_total / max(1, len(train_loader)):.6f}, "
            f"validation={validation_loss:.6f}"
        )

    return checkpoint_path


def evaluate_esrgan(generator: object, loader: object, device: object) -> float:
    """Evaluate average L1 loss without retaining validation tensors."""
    torch, _ = _load_torch_training_tools()
    generator.eval()
    loss_function = torch.nn.L1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for lr_images, hr_images in loader:
            predictions = generator(lr_images.to(device))
            total_loss += float(
                loss_function(predictions, hr_images.to(device)).cpu()
            )
    return total_loss / max(1, len(loader))


def enhance_thermal_image(
    thermal_path: Path,
    checkpoint_path: Path,
    output_path: Path | None = None,
    tile_size: int = 512,
    overlap: int = 32,
    device: str | None = None,
    mixed_precision: bool = True,
) -> Path:
    """Run overlapped tiled inference and write one complete GeoTIFF.

    Each tile includes contextual pixels around its output core. Only the
    core is written, so adjacent model predictions cannot create hard seams.
    """
    torch, _ = _load_torch_training_tools()
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and overlap must be smaller than tile_size.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator, _ = create_esrgan_models(
        scale_factor=int(checkpoint.get("scale_factor", SCALE_FACTOR)),
        feature_channels=int(checkpoint.get("feature_channels", 64)),
        rrdb_blocks=int(checkpoint.get("rrdb_blocks", 23)),
    )
    selected_device = _select_device(torch, device)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(selected_device)
    generator.eval()
    if selected_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_mixed_precision = mixed_precision and selected_device.type == "cuda"
    scale_factor = int(checkpoint.get("scale_factor", SCALE_FACTOR))

    if output_path is None:
        output_path = thermal_path.parent.parent / "Enhanced_Thermal" / "Enhanced_Thermal_Image.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(thermal_path) as source:
        output_width = source.width * scale_factor
        output_height = source.height * scale_factor
        total_tiles = (
            (source.height + tile_size - 1) // tile_size
        ) * ((source.width + tile_size - 1) // tile_size)
        processed_tiles = 0
        print(
            f"Enhancing {source.width} x {source.height} image "
            f"({total_tiles} tiles)...",
            flush=True,
        )
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            width=output_width,
            height=output_height,
            count=1,
            dtype="uint8",
            compress="lzw",
            transform=source.transform * Affine.scale(
                1 / scale_factor, 1 / scale_factor
            ),
        )
        with rasterio.open(output_path, "w", **profile) as destination:
            for top in range(0, source.height, tile_size):
                for left in range(0, source.width, tile_size):
                    height = min(tile_size, source.height - top)
                    width = min(tile_size, source.width - left)
                    read_left = max(0, left - overlap)
                    read_top = max(0, top - overlap)
                    read_right = min(source.width, left + width + overlap)
                    read_bottom = min(source.height, top + height + overlap)
                    read_width = read_right - read_left
                    read_height = read_bottom - read_top
                    tile = source.read(
                        1,
                        window=Window(
                            read_left,
                            read_top,
                            read_width,
                            read_height,
                        ),
                    )
                    pad_left = read_left - (left - overlap)
                    pad_top = read_top - (top - overlap)
                    pad_right = (left + width + overlap) - read_right
                    pad_bottom = (top + height + overlap) - read_bottom
                    if any((pad_left, pad_top, pad_right, pad_bottom)):
                        tile = cv2.copyMakeBorder(
                            tile,
                            pad_top,
                            pad_bottom,
                            pad_left,
                            pad_right,
                            cv2.BORDER_REFLECT_101,
                        )
                    input_tensor = (
                        torch.from_numpy(tile.copy())
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(selected_device)
                        / 255.0
                    )
                    with torch.inference_mode(), torch.autocast(
                        device_type=selected_device.type,
                        dtype=torch.float16,
                        enabled=use_mixed_precision,
                    ):
                        enhanced = generator(input_tensor).clamp(0, 1)
                    crop_top = overlap * scale_factor
                    crop_left = overlap * scale_factor
                    enhanced_tile = enhanced[
                        0,
                        0,
                        crop_top:crop_top + height * scale_factor,
                        crop_left:crop_left + width * scale_factor,
                    ]
                    enhanced_tile = (
                        enhanced_tile.mul(255.0)
                        .round()
                        .to(dtype=torch.uint8)
                        .cpu()
                        .numpy()
                    )
                    del input_tensor, enhanced, tile
                    enhanced_tile = np.ascontiguousarray(enhanced_tile)
                    destination.write(
                        enhanced_tile,
                        indexes=1,
                        window=Window(
                            left * scale_factor,
                            top * scale_factor,
                            width * scale_factor,
                            height * scale_factor,
                        ),
                    )
                    del enhanced_tile
                    processed_tiles += 1
                    if processed_tiles % 100 == 0:
                        if selected_device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                    if processed_tiles == 1 or processed_tiles % 100 == 0 or processed_tiles == total_tiles:
                        print(
                            f"Tile progress: {processed_tiles}/{total_tiles}",
                            flush=True,
                        )

    print(f"Enhanced image saved: {output_path}")
    return output_path


def show_thermal_comparison(
    original_path: Path,
    enhanced_path: Path,
    preview_size: int = 1200,
    comparison_path: Path | None = None,
) -> None:
    """Save and display original and enhanced thermal images with dimensions."""
    if preview_size <= 0:
        raise ValueError("preview_size must be positive.")
    if comparison_path is None:
        comparison_path = enhanced_path.parent / "Thermal_Before_After_Comparison.png"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    def read_preview(path: Path) -> tuple[np.ndarray, int, int]:
        with rasterio.open(path) as source:
            width, height = source.width, source.height
            preview_scale = min(1.0, preview_size / max(width, height))
            preview_width = max(1, round(width * preview_scale))
            preview_height = max(1, round(height * preview_scale))
            image = source.read(
                1,
                out_shape=(1, preview_height, preview_width),
                resampling=rasterio.enums.Resampling.bilinear,
            )
        return image, width, height

    original, original_width, original_height = read_preview(original_path)
    enhanced, enhanced_width, enhanced_height = read_preview(enhanced_path)

    figure, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    for axis, image, title, width, height in (
        (
            axes[0],
            original,
            "Before ESRGAN",
            original_width,
            original_height,
        ),
        (
            axes[1],
            enhanced,
            "After ESRGAN",
            enhanced_width,
            enhanced_height,
        ),
    ):
        display_image = axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"{title}\n"
            f"Size: {width} x {height} pixels\n"
            f"Pixel range: {image.min():.1f} - {image.max():.1f} | "
            f"Mean: {image.mean():.1f}",
            fontsize=11,
        )
        axis.set_xlabel("Pixel X")
        axis.set_ylabel("Pixel Y")
        figure.colorbar(display_image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        "Thermal Image Comparison (preview pixels shown with bilinear resampling)",
        fontsize=14,
    )
    figure.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"Comparison preview saved: {comparison_path}")
    plt.show()


@dataclass(frozen=True)
class PatchRecord:
    """Metadata for one paired ESRGAN patch."""

    patch_id: str
    split: str
    hr_path: str
    lr_path: str
    hr_size: list[int]
    lr_size: list[int]


def find_thermal_image(base_path: Path) -> Path:
    """Locate the Step 02 grayscale training image below ``base_path``."""
    candidates = sorted(
        path
        for path in base_path.rglob("Thermal_Image.png")
        if path.is_file()
    )

    if not candidates:
        raise FileNotFoundError(
            f"Thermal_Image.png was not found below: {base_path}\n"
            "Run thermal_preprocessing.py before preparing the ESRGAN dataset."
        )

    if len(candidates) > 1:
        print(
            f"Multiple thermal images found; using the first in sorted order: "
            f"{candidates[0]}"
        )

    return candidates[0]


def generate_patch_windows(
    width: int,
    height: int,
    patch_size: int,
) -> Iterator[tuple[int, Window]]:
    """Yield complete, non-overlapping windows and stable numeric IDs."""
    patch_id = 0
    for top in range(0, height - patch_size + 1, patch_size):
        for left in range(0, width - patch_size + 1, patch_size):
            yield patch_id, Window(left, top, patch_size, patch_size)
            patch_id += 1


def split_patch_ids(
    patch_ids: list[int],
    seed: int,
) -> dict[int, str]:
    """Create a deterministic train/validation/test assignment."""
    shuffled_ids = patch_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)

    train_end = int(len(shuffled_ids) * TRAIN_RATIO)
    validation_end = train_end + int(len(shuffled_ids) * VALIDATION_RATIO)

    assignments: dict[int, str] = {}
    for index, patch_id in enumerate(shuffled_ids):
        if index < train_end:
            assignments[patch_id] = "train"
        elif index < validation_end:
            assignments[patch_id] = "validation"
        else:
            assignments[patch_id] = "test"
    return assignments


def create_output_directories(dataset_root: Path) -> dict[str, tuple[Path, Path]]:
    """Create the paired ESRGAN directory structure."""
    directories: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "validation", "test"):
        hr_directory = dataset_root / split / "HR"
        lr_directory = dataset_root / split / "LR"
        hr_directory.mkdir(parents=True, exist_ok=True)
        lr_directory.mkdir(parents=True, exist_ok=True)
        directories[split] = (hr_directory, lr_directory)
    return directories


def save_preview(
    preview_records: list[PatchRecord],
    dataset_root: Path,
) -> None:
    """Save a visual comparison of several generated LR-HR pairs."""
    if not preview_records:
        return

    figure, axes = plt.subplots(
        len(preview_records),
        2,
        figsize=(8, max(3, 3 * len(preview_records))),
        squeeze=False,
    )

    for row, record in enumerate(preview_records):
        hr_image = cv2.imread(
            str(dataset_root / record.hr_path),
            cv2.IMREAD_GRAYSCALE,
        )
        lr_image = cv2.imread(
            str(dataset_root / record.lr_path),
            cv2.IMREAD_GRAYSCALE,
        )
        if hr_image is None or lr_image is None:
            raise RuntimeError(f"Unable to read preview patch: {record.patch_id}")

        axes[row][0].imshow(lr_image, cmap="gray", vmin=0, vmax=255)
        axes[row][0].set_title(f"{record.split}: LR ({lr_image.shape[1]}x{lr_image.shape[0]})")
        axes[row][1].imshow(hr_image, cmap="gray", vmin=0, vmax=255)
        axes[row][1].set_title(f"{record.split}: HR ({hr_image.shape[1]}x{hr_image.shape[0]})")
        axes[row][0].axis("off")
        axes[row][1].axis("off")

    figure.tight_layout()
    figure.savefig(dataset_root / "LR_HR_Preview.png", dpi=150)
    plt.close(figure)


def prepare_esrgan_dataset(
    base_path: Path | None = None,
    patch_size: int = PATCH_SIZE,
    scale_factor: int = SCALE_FACTOR,
    seed: int = RANDOM_SEED,
    preview_count: int = PREVIEW_COUNT,
) -> Path:
    """Generate paired ESRGAN patches and return the dataset directory."""
    if patch_size <= 0 or scale_factor <= 0:
        raise ValueError("patch_size and scale_factor must be positive.")
    if patch_size % scale_factor != 0:
        raise ValueError("patch_size must be divisible by scale_factor.")
    if not np.isclose(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO, 1.0):
        raise ValueError("Dataset split ratios must add up to 1.0.")

    selected_base_path = (
        base_path if base_path is not None else select_scene_directory(DEFAULT_BASE_PATH)
    )
    thermal_path = find_thermal_image(selected_base_path)
    dataset_root = thermal_path.parent.parent / "ESRGAN_Dataset"
    directories = create_output_directories(dataset_root)
    metadata: list[PatchRecord] = []

    with rasterio.open(thermal_path) as source:
        if source.count != 1:
            raise ValueError(f"Expected a single-channel thermal image: {thermal_path}")

        patch_windows = list(
            generate_patch_windows(source.width, source.height, patch_size)
        )
        assignments = split_patch_ids(
            [patch_id for patch_id, _ in patch_windows],
            seed,
        )

        print(f"Source image : {thermal_path}")
        print(f"Source size  : {source.width} x {source.height}")
        print(f"Complete patches: {len(patch_windows)}")

        for patch_id, window in patch_windows:
            split = assignments[patch_id]
            hr_directory, lr_directory = directories[split]
            patch_name = f"thermal_{patch_id:06d}.png"
            hr_path = hr_directory / patch_name
            lr_path = lr_directory / patch_name

            hr_patch = source.read(1, window=window)
            if hr_patch.dtype != np.uint8:
                hr_patch = np.clip(hr_patch, 0, 255).astype(np.uint8)

            lr_size = patch_size // scale_factor
            lr_patch = cv2.resize(
                hr_patch,
                (lr_size, lr_size),
                interpolation=cv2.INTER_CUBIC,
            )

            if not cv2.imwrite(str(hr_path), hr_patch):
                raise OSError(f"Failed to write HR patch: {hr_path}")
            if not cv2.imwrite(str(lr_path), lr_patch):
                raise OSError(f"Failed to write LR patch: {lr_path}")

            metadata.append(
                PatchRecord(
                    patch_id=patch_name.removesuffix(".png"),
                    split=split,
                    hr_path=str(hr_path.relative_to(dataset_root)),
                    lr_path=str(lr_path.relative_to(dataset_root)),
                    hr_size=[patch_size, patch_size],
                    lr_size=[lr_size, lr_size],
                )
            )
            del hr_patch, lr_patch
            gc.collect()

    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_image": str(thermal_path),
                "patch_size": patch_size,
                "scale_factor": scale_factor,
                "lr_downsampling": "bicubic",
                "split_ratios": {
                    "train": TRAIN_RATIO,
                    "validation": VALIDATION_RATIO,
                    "test": TEST_RATIO,
                },
                "seed": seed,
                "total_patches": len(metadata),
                "patches_by_split": {
                    split: sum(record.split == split for record in metadata)
                    for split in ("train", "validation", "test")
                },
                "patches": [asdict(record) for record in metadata],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_preview(metadata[:preview_count], dataset_root)
    print(f"Dataset created: {dataset_root}")
    print(f"Metadata saved : {metadata_path}")
    return dataset_root


if __name__ == "__main__":
    prepare_esrgan_dataset()
