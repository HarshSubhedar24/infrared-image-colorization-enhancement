"""Create aligned thermal/RGB patches for Pix2Pix training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

from scene_selection import select_scene_directory


PATCH_SIZE = 256
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42
MIN_VALID_FRACTION = 0.80
PREVIEW_COUNT = 5


def prepare_pix2pix_dataset(
    thermal_path: Path,
    rgb_path: Path,
    output_root: Path,
    patch_size: int = PATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> Path:
    """Create synchronized thermal and RGB PNG patches."""
    if not thermal_path.is_file():
        raise FileNotFoundError(f"Aligned thermal raster not found: {thermal_path}")
    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    with rasterio.open(thermal_path) as thermal, rasterio.open(rgb_path) as rgb:
        if thermal.count != 1:
            raise ValueError("The aligned thermal raster must contain one band.")
        if rgb.count != 3:
            raise ValueError("The RGB image must contain three bands.")
        if (thermal.width, thermal.height) != (rgb.width, rgb.height):
            raise ValueError(
                "Thermal and RGB dimensions do not match: "
                f"{thermal.width}x{thermal.height} versus {rgb.width}x{rgb.height}."
            )

        candidates = [
            (left, top)
            for top in range(0, thermal.height - patch_size + 1, patch_size)
            for left in range(0, thermal.width - patch_size + 1, patch_size)
        ]
        if not candidates:
            raise ValueError("The images are smaller than the requested patch size.")

        valid_candidates = []
        for left, top in candidates:
            window = Window(left, top, patch_size, patch_size)
            thermal_patch = thermal.read(1, window=window)
            rgb_patch = rgb.read(window=window)
            valid_fraction = _valid_fraction(thermal_patch, rgb_patch)
            if valid_fraction >= MIN_VALID_FRACTION:
                valid_candidates.append((left, top))

        if not valid_candidates:
            raise ValueError(
                "No valid paired patches were found. Check alignment and NoData values."
            )

        rng = random.Random(seed)
        rng.shuffle(valid_candidates)
        splits = _split_candidates(valid_candidates)
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "thermal_source": str(thermal_path),
            "rgb_source": str(rgb_path),
            "patch_size": patch_size,
            "seed": seed,
            "minimum_valid_fraction": MIN_VALID_FRACTION,
            "counts": {name: len(items) for name, items in splits.items()},
            "splits": {},
        }

        for split_name, split_candidates in splits.items():
            thermal_directory = output_root / split_name / "thermal"
            rgb_directory = output_root / split_name / "rgb"
            thermal_directory.mkdir(parents=True, exist_ok=True)
            rgb_directory.mkdir(parents=True, exist_ok=True)
            manifest["splits"][split_name] = []

            for patch_number, (left, top) in enumerate(split_candidates):
                window = Window(left, top, patch_size, patch_size)
                thermal_patch = thermal.read(1, window=window)
                rgb_patch = np.transpose(rgb.read(window=window), (1, 2, 0))
                patch_name = f"patch_{patch_number:06d}_x{left}_y{top}.png"
                Image.fromarray(thermal_patch.astype(np.uint8), mode="L").save(
                    thermal_directory / patch_name
                )
                Image.fromarray(rgb_patch.astype(np.uint8), mode="RGB").save(
                    rgb_directory / patch_name
                )
                manifest["splits"][split_name].append(
                    {"name": patch_name, "left": left, "top": top}
                )

            _write_preview(output_root, split_name, thermal_directory, rgb_directory)

        (output_root / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

    print(f"Pix2Pix dataset saved: {output_root}")
    print(f"Valid paired patches: {len(valid_candidates)}")
    for split_name, items in splits.items():
        print(f"  {split_name}: {len(items)}")
    return output_root


def _valid_fraction(thermal_patch: np.ndarray, rgb_patch: np.ndarray) -> float:
    thermal_valid = thermal_patch > 0
    rgb_valid = np.any(rgb_patch > 0, axis=0)
    return float(np.mean(thermal_valid & rgb_valid))


def _split_candidates(candidates: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    total = len(candidates)
    if total < 3:
        raise ValueError("At least three valid patches are required for dataset splits.")
    train_end = int(total * TRAIN_RATIO)
    validation_end = train_end + int(total * VALIDATION_RATIO)
    train_end = min(max(1, train_end), total - 2)
    validation_end = max(train_end + 1, validation_end)
    validation_end = min(validation_end, total - 1)
    if total - validation_end == 0:
        validation_end -= 1
    return {
        "train": candidates[:train_end],
        "validation": candidates[train_end:validation_end],
        "test": candidates[validation_end:],
    }


def _write_preview(
    output_root: Path,
    split_name: str,
    thermal_directory: Path,
    rgb_directory: Path,
) -> None:
    thermal_files = sorted(thermal_directory.glob("*.png"))[:PREVIEW_COUNT]
    if not thermal_files:
        return
    canvas = Image.new("RGB", (PATCH_SIZE * 2, PATCH_SIZE * len(thermal_files)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, thermal_file in enumerate(thermal_files):
        rgb_file = rgb_directory / thermal_file.name
        thermal_image = Image.open(thermal_file).convert("RGB")
        rgb_image = Image.open(rgb_file).convert("RGB")
        canvas.paste(thermal_image, (0, index * PATCH_SIZE))
        canvas.paste(rgb_image, (PATCH_SIZE, index * PATCH_SIZE))
        draw.text((5, index * PATCH_SIZE + 5), "Thermal", fill="yellow")
        draw.text((PATCH_SIZE + 5, index * PATCH_SIZE + 5), "RGB", fill="yellow")
    canvas.save(output_root / f"{split_name}_paired_preview.png")


def main() -> None:
    scene_directory = select_scene_directory()
    output_directory = scene_directory / "Output"
    prepare_pix2pix_dataset(
        output_directory / "Enhanced_Thermal" / "Enhanced_Thermal_Pix2Pix.tif",
        output_directory / "RGB_Image.png",
        output_directory / "Pix2Pix_Dataset",
    )


if __name__ == "__main__":
    main()
