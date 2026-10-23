"""Create one scene-level-split Pix2Pix dataset from multiple Landsat scenes."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

from scene_selection import select_scene_root_directory


PATCH_SIZE = 256
MIN_VALID_FRACTION = 0.80
MIN_SCENES = 10
TRAIN_SCENE_RATIO = 0.80
VALIDATION_SCENE_RATIO = 0.10
RANDOM_SEED = 42
PREVIEW_COUNT = 5
PROJECT_DATASET_ROOT = Path(__file__).resolve().parent / "Pix2Pix_Combined_Dataset"
DEFAULT_DATASET_ROOT = Path(r"D:\datasets\Pix2Pix_Combined_Dataset")


def default_output_root() -> Path:
    """Resolve the shared dataset output location."""
    configured_path = os.environ.get("PIX2PIX_DATASET_ROOT")
    if configured_path:
        return Path(configured_path).expanduser()
    if DEFAULT_DATASET_ROOT.parent.is_dir():
        return DEFAULT_DATASET_ROOT
    return PROJECT_DATASET_ROOT


def prepare_combined_pix2pix_dataset(
    scenes_root: Path,
    output_root: Path,
    patch_size: int = PATCH_SIZE,
    seed: int = RANDOM_SEED,
    minimum_scenes: int = MIN_SCENES,
) -> Path:
    """Build paired patches from all valid scene directories below ``scenes_root``."""
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if minimum_scenes < 3:
        raise ValueError("minimum_scenes must be at least 3.")

    scene_records = []
    for scene_directory in sorted(path for path in scenes_root.iterdir() if path.is_dir()):
        thermal_path = (
            scene_directory
            / "Output"
            / "Enhanced_Thermal"
            / "Enhanced_Thermal_Pix2Pix.tif"
        )
        rgb_path = scene_directory / "Output" / "RGB_Image.png"
        if not thermal_path.is_file() or not rgb_path.is_file():
            continue
        valid_candidates = _find_valid_candidates(
            thermal_path, rgb_path, patch_size
        )
        if valid_candidates:
            scene_records.append(
                {
                    "name": scene_directory.name,
                    "directory": scene_directory,
                    "thermal_path": thermal_path,
                    "rgb_path": rgb_path,
                    "candidates": valid_candidates,
                }
            )

    if len(scene_records) < minimum_scenes:
        raise ValueError(
            f"At least {minimum_scenes} valid scenes are required, but only "
            f"{len(scene_records)} were found below {scenes_root}. Each scene "
            "must contain aligned thermal and RGB outputs."
        )

    rng = random.Random(seed)
    rng.shuffle(scene_records)
    scene_splits = _split_scenes(scene_records)
    if output_root.exists():
        if output_root.is_file():
            raise FileExistsError(f"Dataset output is a file: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    manifest = {
        "source_root": str(scenes_root),
        "output_root": str(output_root),
        "patch_size": patch_size,
        "minimum_valid_fraction": MIN_VALID_FRACTION,
        "minimum_scenes": minimum_scenes,
        "seed": seed,
        "scene_counts": {name: len(items) for name, items in scene_splits.items()},
        "scenes": {},
        "counts": {},
        "input_channels": [
            "visible_B2_blue",
            "visible_B3_green",
            "visible_B4_red",
            "enhanced_thermal",
        ],
        "target": "RGB_Image",
    }

    for split_name, split_records in scene_splits.items():
        manifest["scenes"][split_name] = []
        patch_number = 0
        for record in split_records:
            manifest["scenes"][split_name].append(record["name"])
            split_count = 0
            with rasterio.open(record["thermal_path"]) as thermal, rasterio.open(
                record["rgb_path"]
            ) as rgb:
                for left, top in record["candidates"]:
                    window = Window(left, top, patch_size, patch_size)
                    thermal_patch = thermal.read(1, window=window)
                    rgb_patch = np.transpose(rgb.read(window=window), (1, 2, 0))
                    patch_name = (
                        f"{record['name']}_patch_{patch_number:07d}_"
                        f"x{left}_y{top}.png"
                    )
                    thermal_directory = output_root / split_name / "input_thermal"
                    visible_directory = output_root / split_name / "input_rgb"
                    rgb_directory = output_root / split_name / "rgb"
                    thermal_directory.mkdir(parents=True, exist_ok=True)
                    visible_directory.mkdir(parents=True, exist_ok=True)
                    rgb_directory.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(thermal_patch.astype(np.uint8), mode="L").save(
                        thermal_directory / patch_name
                    )
                    Image.fromarray(rgb_patch.astype(np.uint8), mode="RGB").save(
                        visible_directory / patch_name
                    )
                    Image.fromarray(rgb_patch.astype(np.uint8), mode="RGB").save(
                        rgb_directory / patch_name
                    )
                    patch_number += 1
                    split_count += 1
            record["split_count"] = split_count
        manifest["counts"][split_name] = patch_number
        _write_preview(output_root, split_name)

    manifest["scene_patch_counts"] = {
        split_name: {
            record["name"]: record["split_count"]
            for record in split_records
        }
        for split_name, split_records in scene_splits.items()
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Combined Pix2Pix dataset saved: {output_root}")
    print(f"Scenes used: {len(scene_records)}")
    for split_name, records in scene_splits.items():
        print(
            f"  {split_name}: {len(records)} scenes, "
            f"{manifest['counts'][split_name]} patches"
        )
    return output_root


def _find_valid_candidates(
    thermal_path: Path, rgb_path: Path, patch_size: int
) -> list[tuple[int, int]]:
    with rasterio.open(thermal_path) as thermal:
        if thermal.count != 1:
            raise ValueError(f"Thermal raster must contain one band: {thermal_path}")
        thermal_width, thermal_height = thermal.width, thermal.height
        with Image.open(rgb_path) as rgb:
            if rgb.size != (thermal_width, thermal_height):
                raise ValueError(
                    f"Thermal and RGB dimensions do not match for {thermal_path.parent.parent.parent}: "
                    f"{thermal_width}x{thermal_height} versus {rgb.width}x{rgb.height}."
                )
            rgb_array = np.asarray(rgb.convert("RGB"))
        candidates = []
        for top in range(0, thermal_height - patch_size + 1, patch_size):
            for left in range(0, thermal_width - patch_size + 1, patch_size):
                window = Window(left, top, patch_size, patch_size)
                thermal_patch = thermal.read(1, window=window)
                left = int(window.col_off)
                top = int(window.row_off)
                rgb_patch = np.transpose(
                    rgb_array[top:top + patch_size, left:left + patch_size],
                    (2, 0, 1),
                )
                if _valid_fraction(thermal_patch, rgb_patch) >= MIN_VALID_FRACTION:
                    candidates.append((left, top))
    return candidates


def _valid_fraction(thermal_patch: np.ndarray, rgb_patch: np.ndarray) -> float:
    thermal_valid = thermal_patch > 0
    rgb_valid = np.any(rgb_patch > 0, axis=0)
    return float(np.mean(thermal_valid & rgb_valid))


def _split_scenes(records: list[dict]) -> dict[str, list[dict]]:
    total = len(records)
    train_count = max(1, int(total * TRAIN_SCENE_RATIO))
    validation_count = max(1, int(total * VALIDATION_SCENE_RATIO))
    if train_count + validation_count >= total:
        validation_count = 1
        train_count = total - 2
    return {
        "train": records[:train_count],
        "validation": records[train_count:train_count + validation_count],
        "test": records[train_count + validation_count:],
    }


def _write_preview(output_root: Path, split_name: str) -> None:
    thermal_directory = output_root / split_name / "input_thermal"
    visible_directory = output_root / split_name / "input_rgb"
    rgb_directory = output_root / split_name / "rgb"
    thermal_files = sorted(thermal_directory.glob("*.png"))[:PREVIEW_COUNT]
    if not thermal_files:
        return
    canvas = Image.new("RGB", (PATCH_SIZE * 2, PATCH_SIZE * len(thermal_files)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, thermal_file in enumerate(thermal_files):
        rgb_file = rgb_directory / thermal_file.name
        visible_file = visible_directory / thermal_file.name
        with Image.open(thermal_file) as thermal_image, Image.open(rgb_file) as rgb_image:
            canvas.paste(thermal_image.convert("RGB"), (0, index * PATCH_SIZE))
            canvas.paste(rgb_image.convert("RGB"), (PATCH_SIZE, index * PATCH_SIZE))
        draw.text((5, index * PATCH_SIZE + 5), "Thermal", fill="yellow")
        draw.text((PATCH_SIZE + 5, index * PATCH_SIZE + 5), "RGB", fill="yellow")
    canvas.save(output_root / f"{split_name}_paired_preview.png")


def main() -> None:
    scenes_root = select_scene_root_directory()
    output_root = default_output_root()
    prepare_combined_pix2pix_dataset(scenes_root, output_root)


if __name__ == "__main__":
    main()
