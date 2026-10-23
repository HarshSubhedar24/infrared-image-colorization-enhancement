"""Generate a colorized RGB image from one scene using the shared Pix2Pix model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from pix2pix_model import create_pix2pix_models
from scene_selection import select_scene_directory


PROJECT_DATASET_ROOT = Path(__file__).resolve().parent / "Pix2Pix_Combined_Dataset"
DEFAULT_DATASET_ROOT = Path(r"D:\datasets\Pix2Pix_Combined_Dataset")
TILE_SIZE = 256
OVERLAP = 64
BLEND_FLOOR = 0.05
VISIBLE_PRESERVATION = 0.70


def find_pix2pix_checkpoint() -> Path:
    """Find the shared best checkpoint using configurable locations."""
    configured_path = os.environ.get("PIX2PIX_CHECKPOINT_PATH")
    candidates = (
        [Path(configured_path).expanduser()]
        if configured_path
        else [
            DEFAULT_DATASET_ROOT / "checkpoints" / "pix2pix_best.pt",
            PROJECT_DATASET_ROOT / "checkpoints" / "pix2pix_best.pt",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Shared Pix2Pix checkpoint was not found. Searched:\n"
        f"{searched}\nSet PIX2PIX_CHECKPOINT_PATH to pix2pix_best.pt."
    )


def run_pix2pix_inference(
    thermal_path: Path,
    visible_rgb_path: Path,
    checkpoint_path: Path,
    output_png: Path,
    output_tif: Path | None = None,
    tile_size: int = TILE_SIZE,
    overlap: int = OVERLAP,
    visible_preservation: float = VISIBLE_PRESERVATION,
    device: str | None = None,
) -> Path:
    """Run tiled Pix2Pix inference and save an RGB PNG and optional GeoTIFF."""
    if not thermal_path.is_file():
        raise FileNotFoundError(f"Aligned thermal raster not found: {thermal_path}")
    if not visible_rgb_path.is_file():
        raise FileNotFoundError(f"Visible RGB image not found: {visible_rgb_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pix2Pix checkpoint not found: {checkpoint_path}")
    if tile_size <= 0 or tile_size % 256 != 0:
        raise ValueError("tile_size must be a positive multiple of 256.")
    if overlap < 0 or overlap >= tile_size // 2:
        raise ValueError("overlap must be non-negative and less than half the tile size.")
    if not 0.0 <= visible_preservation <= 1.0:
        raise ValueError("visible_preservation must be between 0 and 1.")

    import torch

    selected_device = _select_device(torch, device)
    checkpoint = torch.load(checkpoint_path, map_location=selected_device)
    configuration = checkpoint.get("configuration", {})
    if int(configuration.get("input_channels", 1)) != 4:
        raise ValueError(
            "The checkpoint is for the previous one-channel Pix2Pix model. "
            "Retrain using the four-channel fusion dataset first."
        )
    base_channels = int(configuration.get("base_channels", 64))
    generator, _ = create_pix2pix_models(
        input_channels=4,
        output_channels=3,
        base_channels=base_channels,
    )
    generator.load_state_dict(checkpoint["generator"])
    generator.to(selected_device)
    generator.eval()

    stride = tile_size - overlap
    with rasterio.open(thermal_path) as source:
        if source.count != 1:
            raise ValueError("The aligned thermal raster must contain one band.")
        height, width = source.height, source.width
        with Image.open(visible_rgb_path) as visible_image:
            visible_array = np.asarray(visible_image.convert("RGB"))
        if visible_array.shape[:2] != (height, width):
            raise ValueError(
                "Aligned thermal and visible RGB dimensions do not match: "
                f"{width}x{height} versus "
                f"{visible_array.shape[1]}x{visible_array.shape[0]}."
            )
        output = np.zeros((3, height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        valid_pixels = np.zeros((height, width), dtype=bool)
        blend_weights = _create_blend_weights(tile_size, overlap)
        row_positions = _positions(height, tile_size, stride)
        column_positions = _positions(width, tile_size, stride)

        for tile_number, (top, left) in enumerate(
            (
                (top, left)
                for top in row_positions
                for left in column_positions
            ),
            start=1,
        ):
            window = Window(
                left,
                top,
                min(tile_size, width - left),
                min(tile_size, height - top),
            )
            thermal_tile = source.read(1, window=window).astype(np.float32)
            tile_height, tile_width = thermal_tile.shape
            valid_tile = thermal_tile > 0
            padded = _pad_tile(thermal_tile, tile_size)
            visible_tile = _pad_rgb_tile(
                visible_array[top:top + tile_height, left:left + tile_width],
                tile_size,
            )
            visible_tensor = (
                torch.from_numpy(visible_tile.copy()).float().permute(2, 0, 1)
            )
            thermal_tensor = torch.from_numpy(padded.copy()).float().unsqueeze(0)
            input_tensor = (
                torch.cat((visible_tensor, thermal_tensor), dim=0).unsqueeze(0)
                .to(selected_device)
                / 127.5
                - 1.0
            )
            with torch.inference_mode():
                generated = generator(input_tensor)[0].detach().cpu().numpy()
            generated = np.clip((generated + 1.0) * 127.5, 0, 255)
            generated = generated[:, :tile_height, :tile_width]
            visible_output = np.transpose(
                visible_array[top:top + tile_height, left:left + tile_width],
                (2, 0, 1),
            ).astype(np.float32)
            generated = (
                visible_preservation * visible_output
                + (1.0 - visible_preservation) * generated
            )
            tile_weights = blend_weights[:tile_height, :tile_width]
            row_slice = slice(top, top + tile_height)
            column_slice = slice(left, left + tile_width)
            output[:, row_slice, column_slice] += generated * tile_weights
            weights[row_slice, column_slice] += tile_weights
            valid_pixels[row_slice, column_slice] |= valid_tile
            if tile_number % 100 == 0 or tile_number == len(row_positions) * len(column_positions):
                print(
                    f"Pix2Pix inference: tile {tile_number}/"
                    f"{len(row_positions) * len(column_positions)}"
                )

        output = output / np.maximum(weights, 1e-6)
        visible_valid = np.any(visible_array > 0, axis=2)
        visible_fallback = (~valid_pixels) & visible_valid
        output[:, visible_fallback] = np.transpose(
            visible_array[visible_fallback],
            (1, 0),
        )
        output[:, ~(valid_pixels | visible_valid)] = 0
        output = np.rint(np.clip(output, 0, 255)).astype(np.uint8)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.transpose(output, (1, 2, 0)), mode="RGB").save(output_png)
        if output_tif is not None:
            _write_geotiff(output_tif, source, output)

    print(f"Colorized RGB PNG saved: {output_png}")
    if output_tif is not None:
        print(f"Colorized RGB GeoTIFF saved: {output_tif}")
    return output_png


def _select_device(torch: Any, device: str | None) -> Any:
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return torch.device(selected)


def _positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def _pad_tile(tile: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = tile.shape
    pad_height = tile_size - height
    pad_width = tile_size - width
    if pad_height == 0 and pad_width == 0:
        return tile
    return np.pad(
        tile,
        ((0, pad_height), (0, pad_width)),
        mode="reflect" if height > 1 and width > 1 else "edge",
    )


def _pad_rgb_tile(tile: np.ndarray, tile_size: int) -> np.ndarray:
    height, width = tile.shape[:2]
    pad_height = tile_size - height
    pad_width = tile_size - width
    if pad_height == 0 and pad_width == 0:
        return tile
    return np.pad(
        tile,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="reflect" if height > 1 and width > 1 else "edge",
    )


def _create_blend_weights(tile_size: int, overlap: int) -> np.ndarray:
    """Create a raised-cosine window that softly blends neighboring tiles."""
    if overlap == 0:
        return np.ones((tile_size, tile_size), dtype=np.float32)
    ramp_length = min(overlap, tile_size // 2)
    ramp = np.ones(tile_size, dtype=np.float32)
    coordinates = np.arange(ramp_length, dtype=np.float32)
    edge = BLEND_FLOOR + (1.0 - BLEND_FLOOR) * 0.5 * (
        1.0 - np.cos(np.pi * (coordinates + 1.0) / (ramp_length + 1.0))
    )
    ramp[:ramp_length] = edge
    ramp[-ramp_length:] = edge[::-1]
    return np.outer(ramp, ramp)


def _write_geotiff(path: Path, source: rasterio.DatasetReader, output: np.ndarray) -> None:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        count=3,
        dtype="uint8",
        nodata=0,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(output)


def main() -> None:
    configured_scene = os.environ.get("INFRARED_SCENE_DIRECTORY")
    scene_directory = (
        Path(configured_scene).expanduser()
        if configured_scene
        else select_scene_directory()
    )
    if not scene_directory.is_dir():
        raise FileNotFoundError(
            f"Selected scene directory does not exist: {scene_directory}"
        )
    output_directory = scene_directory / "Output"
    thermal_path = (
        output_directory
        / "Enhanced_Thermal"
        / "Enhanced_Thermal_Pix2Pix.tif"
    )
    visible_rgb_path = output_directory / "RGB_Image.png"
    output_png = output_directory / "Pix2Pix_Colorized_RGB.png"
    output_tif = output_directory / "Pix2Pix_Colorized_RGB.tif"
    checkpoint_path = find_pix2pix_checkpoint()
    print(f"Using shared Pix2Pix checkpoint: {checkpoint_path}")
    run_pix2pix_inference(
        thermal_path=thermal_path,
        visible_rgb_path=visible_rgb_path,
        checkpoint_path=checkpoint_path,
        output_png=output_png,
        output_tif=output_tif,
        visible_preservation=VISIBLE_PRESERVATION,
        device="cuda",
    )


if __name__ == "__main__":
    main()
