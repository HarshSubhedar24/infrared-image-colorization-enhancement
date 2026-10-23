"""Run the complete infrared processing pipeline for one selected scene."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scene_selection import select_scene_directory


STAGES = (
    ("RGB preparation", "rgb_combine.py"),
    ("Thermal preprocessing", "thermal_preprocessing.py"),
    ("ESRGAN inference", "run_esrgan_inference.py"),
    ("Pix2Pix thermal alignment", "align_enhanced_thermal.py"),
    ("Pix2Pix inference", "run_pix2pix_inference.py"),
)


REQUIRED_BANDS = ("SR_B2", "SR_B3", "SR_B4", "ST_B10")


def validate_scene_inputs(scene_directory: Path) -> None:
    """Fail early when a selected scene is not a complete Landsat scene."""
    available_names = [
        file_path.name.upper()
        for file_path in scene_directory.rglob("*")
        if file_path.is_file()
    ]
    missing_bands = [
        band
        for band in REQUIRED_BANDS
        if not any(band in file_name for file_name in available_names)
    ]
    if missing_bands:
        raise FileNotFoundError(
            f"Selected scene has no usable Landsat input files: {scene_directory}\n"
            f"Missing bands: {', '.join(missing_bands)}\n"
            "Add the complete Landsat scene files or select another scene."
        )


def main() -> None:
    scene_directory = select_scene_directory()
    validate_scene_inputs(scene_directory)
    environment = os.environ.copy()
    environment["INFRARED_SCENE_DIRECTORY"] = str(scene_directory)

    project_directory = Path(__file__).resolve().parent
    for stage_name, script_name in STAGES:
        print(f"\n{'=' * 60}\nStarting {stage_name}\n{'=' * 60}")
        subprocess.run(
            [sys.executable, str(project_directory / script_name)],
            cwd=project_directory,
            env=environment,
            check=True,
        )
        print(f"\nCompleted: {stage_name}")

    print(f"\nPipeline completed successfully for {scene_directory}.")


if __name__ == "__main__":
    main()
