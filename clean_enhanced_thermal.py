"""Crop NoData borders and create a north-up enhanced thermal GeoTIFF."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject

from scene_selection import select_scene_directory


def clean_enhanced_thermal(
    enhanced_path: Path,
    reference_path: Path,
    output_path: Path,
    nodata_value: float | None = None,
) -> Path:
    """Crop NoData and warp an enhanced raster to a north-up GeoTIFF."""
    if not enhanced_path.is_file():
        raise FileNotFoundError(f"Enhanced raster not found: {enhanced_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Georeferenced reference raster not found: {reference_path}")

    with rasterio.open(enhanced_path) as enhanced, rasterio.open(reference_path) as reference:
        if enhanced.count != 1:
            raise ValueError("The enhanced thermal raster must contain one band.")
        if reference.crs is None:
            raise ValueError(f"Reference raster has no CRS: {reference_path}")
        if reference.width == 0 or reference.height == 0:
            raise ValueError("Reference raster has invalid dimensions.")

        scale_x = reference.width / enhanced.width
        scale_y = reference.height / enhanced.height
        enhanced_transform = reference.transform * Affine.scale(scale_x, scale_y)
        source_crs = reference.crs
        source_bounds = rasterio.transform.array_bounds(
            enhanced.height,
            enhanced.width,
            enhanced_transform,
        )
        source_nodata = (
            nodata_value
            if nodata_value is not None
            else enhanced.nodata
            if enhanced.nodata is not None
            else 0
        )
        resolution = (
            abs(reference.transform.a) * scale_x,
            abs(reference.transform.e) * scale_y,
        )
        target_transform, target_width, target_height = calculate_default_transform(
            source_crs,
            source_crs,
            enhanced.width,
            enhanced.height,
            *source_bounds,
            resolution=resolution,
        )

        print(
            f"Reprojecting {enhanced.width} x {enhanced.height} raster "
            f"to a north-up grid ({target_width} x {target_height})..."
        )

        temporary_source = _make_georeferenced_source(
            enhanced,
            enhanced_transform,
            source_crs,
            source_nodata,
        )
        try:
            with rasterio.open(temporary_source) as source:
                mask_scale = max(1, int(max(target_width, target_height) / 2048))
                mask_width = max(1, (target_width + mask_scale - 1) // mask_scale)
                mask_height = max(1, (target_height + mask_scale - 1) // mask_scale)
                mask_transform = target_transform * Affine.scale(mask_scale, mask_scale)
                valid_mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
                reproject(
                    source=rasterio.band(source, 1),
                    destination=valid_mask,
                    dst_transform=mask_transform,
                    dst_crs=source_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                # ESRGAN can leave 1-2 intensity values in black padding after
                # contrast stretching; exclude those edge artifacts as NoData.
                valid_threshold = 2 if enhanced.dtypes[0] == "uint8" else source_nodata
                valid_rows, valid_cols = np.nonzero(valid_mask > valid_threshold)
                if valid_rows.size == 0:
                    raise ValueError("No valid pixels were found after NoData masking.")

                row_start = max(0, int(valid_rows.min()) * mask_scale - mask_scale)
                row_stop = min(target_height, int(valid_rows.max() + 1) * mask_scale + mask_scale)
                col_start = max(0, int(valid_cols.min()) * mask_scale - mask_scale)
                col_stop = min(target_width, int(valid_cols.max() + 1) * mask_scale + mask_scale)
                crop_window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)
                cropped_transform = rasterio.windows.transform(crop_window, target_transform)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                profile = enhanced.profile.copy()
                profile.update(driver="GTiff", width=int(crop_window.width), height=int(crop_window.height),
                               count=1, dtype=enhanced.dtypes[0], crs=source_crs,
                               transform=cropped_transform, nodata=0, compress="lzw", BIGTIFF="IF_SAFER")
                if profile["width"] >= 256 and profile["height"] >= 256:
                    profile.update(tiled=True, blockxsize=256, blockysize=256)
                else:
                    profile.pop("tiled", None)
                    profile.pop("blockxsize", None)
                    profile.pop("blockysize", None)

                with rasterio.open(output_path, "w", **profile) as destination:
                    with WarpedVRT(source, transform=cropped_transform,
                                   width=destination.width, height=destination.height,
                                   nodata=0, resampling=Resampling.bilinear) as warped:
                        _write_warped_windows(warped, destination)
                    destination.update_tags(source_file=str(enhanced_path),
                                            processing="NoData border crop and north-up reprojection",
                                            resampling="bilinear")
        finally:
            Path(temporary_source).unlink(missing_ok=True)

    print(f"Clean north-up thermal raster saved: {output_path}")
    return output_path


def _make_georeferenced_source(dataset, transform, crs, nodata):
    """Create a disk-backed source with the corrected geotransform."""
    handle = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    handle.close()
    profile = dataset.profile.copy()
    profile.update(driver="GTiff", crs=crs, transform=transform, nodata=nodata, compress="lzw")
    with rasterio.open(handle.name, "w", **profile) as destination:
        for window in dataset.block_windows(1):
            destination.write(dataset.read(1, window=window[1]), 1, window=window[1])
    return handle.name


def _write_warped_windows(source, destination, window_size=1024):
    for top in range(0, destination.height, window_size):
        for left in range(0, destination.width, window_size):
            window = Window(left, top, min(window_size, destination.width - left),
                            min(window_size, destination.height - top))
            destination.write(source.read(1, window=window), 1, window=window)


def _reproject_to_array_windows(
    source: rasterio.band,
    destination: np.ndarray,
    source_transform: Affine,
    destination_transform: Affine,
    source_crs: object,
    destination_crs: object,
    source_nodata: float,
    destination_nodata: float,
    resampling: Resampling,
    window_size: int = 1024,
) -> None:
    """Reproject a source band into a bounded array window by window."""
    for top in range(0, destination.shape[0], window_size):
        for left in range(0, destination.shape[1], window_size):
            height = min(window_size, destination.shape[0] - top)
            width = min(window_size, destination.shape[1] - left)
            window = Window(left, top, width, height)
            target = np.empty((height, width), dtype=destination.dtype)
            reproject(
                source=source,
                destination=target,
                src_transform=source_transform,
                src_crs=source_crs,
                src_nodata=source_nodata,
                dst_transform=destination_transform * Affine.translation(left, top),
                dst_crs=destination_crs,
                dst_nodata=destination_nodata,
                init_dest_nodata=True,
                resampling=resampling,
            )
            destination[window.row_off : window.row_off + height,
                        window.col_off : window.col_off + width] = target


def _reproject_to_dataset_windows(
    source: rasterio.band,
    destination: rasterio.io.DatasetWriter,
    source_transform: Affine,
    destination_transform: Affine,
    source_crs: object,
    destination_crs: object,
    source_nodata: float,
    destination_nodata: float,
    resampling: Resampling,
    window_size: int = 1024,
) -> None:
    """Reproject a source band directly into tiled output windows."""
    for top in range(0, destination.height, window_size):
        for left in range(0, destination.width, window_size):
            height = min(window_size, destination.height - top)
            width = min(window_size, destination.width - left)
            window = Window(left, top, width, height)
            target = np.empty((height, width), dtype=destination.dtypes[0])
            reproject(
                source=source,
                destination=target,
                src_transform=source_transform,
                src_crs=source_crs,
                src_nodata=source_nodata,
                dst_transform=destination_transform * Affine.translation(left, top),
                dst_crs=destination_crs,
                dst_nodata=destination_nodata,
                init_dest_nodata=True,
                resampling=resampling,
            )
            destination.write(target, 1, window=window)


def main() -> None:
    scene_directory = select_scene_directory()
    output_directory = scene_directory / "Output"
    enhanced_path = output_directory / "Enhanced_Thermal" / "Enhanced_Thermal_Image.tif"
    reference_path = output_directory / "Thermal_TIFF" / "Temperature_Celsius.tif"
    output_path = output_directory / "Enhanced_Thermal" / "Enhanced_Thermal_Clean.tif"
    clean_enhanced_thermal(enhanced_path, reference_path, output_path)


if __name__ == "__main__":
    main()
