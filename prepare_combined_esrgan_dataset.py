"""Build one ESRGAN dataset from the thermal images of multiple scenes."""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

from esrgan_dataset_preparation import (
    PATCH_SIZE,
    SCALE_FACTOR,
    create_output_directories,
    find_thermal_image,
)
from scene_selection import DEFAULT_SCENE_ROOT, select_scene_root_directory


TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42


def prepare_combined_dataset(
    scenes_root: Path = DEFAULT_SCENE_ROOT,
    output_root: Path | None = None,
    patch_size: int = PATCH_SIZE,
    scale_factor: int = SCALE_FACTOR,
    seed: int = RANDOM_SEED,
) -> Path:
    """Create one scene-level-split ESRGAN dataset for all prepared scenes."""
    if patch_size <= 0 or patch_size % scale_factor:
        raise ValueError("patch_size must be positive and divisible by scale_factor.")
    scenes = sorted(path for path in scenes_root.iterdir() if path.is_dir())
    sources = []
    for scene in scenes:
        try:
            sources.append((scene.name, find_thermal_image(scene)))
        except FileNotFoundError:
            print(f"Skipping {scene.name}: thermal preprocessing output not found.")
    if len(sources) < 3:
        raise ValueError("At least three preprocessed scenes are required.")

    random.Random(seed).shuffle(sources)
    train_end = min(max(1, int(len(sources) * TRAIN_RATIO)), len(sources) - 2)
    validation_end = min(
        len(sources) - 1,
        train_end + max(1, int(len(sources) * VALIDATION_RATIO)),
    )
    scene_splits = {
        **{name: "train" for name, _ in sources[:train_end]},
        **{name: "validation" for name, _ in sources[train_end:validation_end]},
        **{name: "test" for name, _ in sources[validation_end:]},
    }
    dataset_root = output_root or Path(__file__).resolve().parent / "ESRGAN_Combined_Dataset"
    directories = create_output_directories(dataset_root)
    metadata = []
    lr_size = patch_size // scale_factor

    for scene_name, thermal_path in sources:
        split = scene_splits[scene_name]
        with rasterio.open(thermal_path) as source:
            windows = [
                Window(left, top, patch_size, patch_size)
                for top in range(0, source.height - patch_size + 1, patch_size)
                for left in range(0, source.width - patch_size + 1, patch_size)
            ]
            for patch_id, window in enumerate(windows):
                hr_patch = source.read(1, window=window)
                hr_patch = np.clip(hr_patch, 0, 255).astype(np.uint8)
                lr_patch = cv2.resize(hr_patch, (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
                name = f"{scene_name}_thermal_{patch_id:06d}.png"
                hr_path = directories[split][0] / name
                lr_path = directories[split][1] / name
                if not cv2.imwrite(str(hr_path), hr_patch) or not cv2.imwrite(str(lr_path), lr_patch):
                    raise OSError(f"Failed to write ESRGAN patch {name}.")
                metadata.append({
                    "scene": scene_name,
                    "split": split,
                    "hr_path": str(hr_path.relative_to(dataset_root)),
                    "lr_path": str(lr_path.relative_to(dataset_root)),
                })
                del hr_patch, lr_patch
                gc.collect()

    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(json.dumps({
        "sources": {name: str(path) for name, path in sources},
        "scene_splits": scene_splits,
        "patch_size": patch_size,
        "scale_factor": scale_factor,
        "seed": seed,
        "total_patches": len(metadata),
        "patches_by_split": {split: sum(item["split"] == split for item in metadata)
                             for split in ("train", "validation", "test")},
        "patches": metadata,
    }, indent=2), encoding="utf-8")
    print(f"Combined ESRGAN dataset created: {dataset_root}")
    print(f"Scenes used: {len(sources)}; patches: {len(metadata)}")
    return dataset_root


if __name__ == "__main__":
    prepare_combined_dataset(select_scene_root_directory())
