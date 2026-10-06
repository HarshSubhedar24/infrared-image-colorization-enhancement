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
    ("ESRGAN dataset preparation", "esrgan_dataset_preparation.py"),
    ("ESRGAN training", "train_esrgan.py"),
    ("ESRGAN inference", "run_esrgan_inference.py"),
    ("GIS cleanup and north-up reprojection", "clean_enhanced_thermal.py"),
)


def confirm_stage(stage_name: str) -> bool:
    answer = input(
        f"\nStart {stage_name}? [Y/n]: "
    ).strip().lower()
    return answer in ("", "y", "yes")


def main() -> None:
    scene_directory = select_scene_directory()
    environment = os.environ.copy()
    environment["INFRARED_SCENE_DIRECTORY"] = str(scene_directory)

    project_directory = Path(__file__).resolve().parent
    for stage_name, script_name in STAGES:
        if not confirm_stage(stage_name):
            print(f"\nPipeline stopped before {stage_name}.")
            return

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
