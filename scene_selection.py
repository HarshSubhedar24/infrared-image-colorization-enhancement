"""Interactive selection of a Landsat scene directory."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_SCENE_ROOT = Path.cwd()


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


def browse_for_directory(initial_directory: Path, title: str) -> Path:
    """Open the Windows folder picker so a folder can be found anywhere."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError(
            "The Windows folder picker requires tkinter. "
            "Install a Python distribution that includes tkinter."
        ) from error

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(
            title=title,
            initialdir=str(initial_directory),
            mustexist=True,
        )
    finally:
        root.destroy()

    if not selected:
        raise ValueError("No scene folder was selected.")
    return Path(selected).expanduser()


def select_scene_root_directory(default_root: Path = DEFAULT_SCENE_ROOT) -> Path:
    """Browse for the parent folder containing multiple scene directories."""
    return browse_for_directory(
        Path.home(),
        "Browse entire device and select the scene parent folder",
    )


def select_scene_directory(
    default_root: Path = DEFAULT_SCENE_ROOT,
) -> Path:
    """Browse the entire device and select one scene folder."""
    selected_from_environment = os.environ.get("INFRARED_SCENE_DIRECTORY")
    if selected_from_environment:
        selected_scene = Path(selected_from_environment).expanduser()
        if not selected_scene.is_dir():
            raise NotADirectoryError(
                f"The selected scene folder does not exist: {selected_scene}"
            )
        print(f"\nUsing selected scene: {selected_scene}")
        return selected_scene

    print("\nBrowse entire device to select a Landsat scene folder.")
    selected_scene = browse_for_directory(
        Path.home(),
        "Browse entire device and select a Landsat scene folder",
    )

    print(f"\nSelected scene: {selected_scene}")
    return selected_scene
