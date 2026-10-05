"""Interactive selection of a Landsat scene directory."""

from __future__ import annotations

import re
import os
from pathlib import Path


DEFAULT_SCENE_ROOT = Path(
    r"D:\datasets"
)


def natural_sort_key(path: Path) -> list[object]:
    """Sort names containing numbers in human-friendly numeric order."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def display_existing_output_files(output_folder: Path) -> None:
    """Report existing output files so a reused scene is clearly identified."""
    if not output_folder.is_dir():
        print(f"\nOutput folder will be created: {output_folder}")
        return

    existing_files = sorted(
        path for path in output_folder.rglob("*") if path.is_file()
    )
    print(f"\nExisting output folder found: {output_folder}")
    if not existing_files:
        print("The output folder is empty.")
        return

    print("Existing output files:")
    for file_path in existing_files:
        print(f"  - {file_path.relative_to(output_folder)}")


def select_scene_directory(
    default_root: Path = DEFAULT_SCENE_ROOT,
) -> Path:
    """Ask the user to select one scene from a directory of scene folders."""
    selected_from_environment = os.environ.get("INFRARED_SCENE_DIRECTORY")
    if selected_from_environment:
        selected_scene = Path(selected_from_environment).expanduser()
        if not selected_scene.is_dir():
            raise NotADirectoryError(
                f"The selected scene folder does not exist: {selected_scene}"
            )
        print(f"\nUsing selected scene: {selected_scene}")
        return selected_scene

    root_input = input(
        f"\nEnter the parent folder containing the scenes "
        f"[{default_root}]: "
    ).strip()
    scene_root = Path(root_input).expanduser() if root_input else default_root

    if not scene_root.is_dir():
        raise NotADirectoryError(
            f"The selected parent folder does not exist: {scene_root}"
        )

    scene_directories = sorted(
        (path for path in scene_root.iterdir() if path.is_dir()),
        key=natural_sort_key,
    )
    if not scene_directories:
        raise FileNotFoundError(
            f"No scene folders were found in: {scene_root}"
        )

    print("\nAvailable scene folders:")
    for index, scene_directory in enumerate(scene_directories, start=1):
        print(f"  {index}. {scene_directory.name}")

    selection = input(
        "\nEnter a scene number, or enter the complete scene path: "
    ).strip()

    if selection.isdigit():
        scene_index = int(selection)
        if not 1 <= scene_index <= len(scene_directories):
            raise ValueError(
                f"Scene number must be between 1 and {len(scene_directories)}."
            )
        selected_scene = scene_directories[scene_index - 1]
    else:
        selected_scene = Path(selection).expanduser()
        if not selected_scene.is_dir():
            raise NotADirectoryError(
                f"The selected scene folder does not exist: {selected_scene}"
            )

    print(f"\nSelected scene: {selected_scene}")
    return selected_scene
