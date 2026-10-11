"""Align the enhanced thermal raster to the original RGB pixel grid."""

from __future__ import annotations

from pathlib import Path
import tempfile

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

from scene_selection import select_scene_directory


def align_enhanced_thermal(
    enhanced_path: Path,
    reference_path: Path,
    output_path: Path,
) -> Path:
    """Create a thermal GeoTIFF with exactly the reference raster grid."""
    if not enhanced_path.is_file():
        raise FileNotFoundError(f"Enhanced raster not found: {enhanced_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference raster not found: {reference_path}")

    with rasterio.open(enhanced_path) as enhanced, rasterio.open(reference_path) as reference:
        if enhanced.count != 1:
            raise ValueError("The enhanced thermal raster must contain one band.")
        if reference.crs is None:
            raise ValueError(f"Reference raster has no CRS: {reference_path}")

        source_nodata = enhanced.nodata if enhanced.nodata is not None else 0
        scale_x = reference.width / enhanced.width
        scale_y = reference.height / enhanced.height
        source_transform = reference.transform * Affine.scale(scale_x, scale_y)
        temporary_source = _make_georeferenced_source(
            enhanced,
            source_transform,
            reference.crs,
            source_nodata,
        )

        try:
            with rasterio.open(temporary_source) as source:
                profile = reference.profile.copy()
                profile.update(
                    driver="GTiff",
                    width=reference.width,
                    height=reference.height,
                    count=1,
                    dtype=enhanced.dtypes[0],
                    crs=reference.crs,
                    transform=reference.transform,
                    nodata=0,
                    compress="lzw",
                    BIGTIFF="IF_SAFER",
                )
                if profile["width"] >= 256 and profile["height"] >= 256:
                    profile.update(tiled=True, blockxsize=256, blockysize=256)

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(output_path, "w", **profile) as destination:
                    with WarpedVRT(
                        source,
                        crs=reference.crs,
                        transform=reference.transform,
                        width=reference.width,
                        height=reference.height,
                        nodata=0,
                        resampling=Resampling.bilinear,
                    ) as warped:
                        for window in _windows(destination.width, destination.height):
                            destination.write(
                                warped.read(1, window=window),
                                1,
                                window=window,
                            )
                    destination.update_tags(
                        source_file=str(enhanced_path),
                        reference_grid=str(reference_path),
                        processing="Aligned to reference RGB grid using bilinear resampling",
                    )
        finally:
            Path(temporary_source).unlink(missing_ok=True)

    print(f"Pix2Pix-aligned thermal raster saved: {output_path}")
    return output_path


def _make_georeferenced_source(dataset, transform, crs, nodata) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    handle.close()
    profile = dataset.profile.copy()
    profile.update(driver="GTiff", crs=crs, transform=transform, nodata=nodata)
    with rasterio.open(handle.name, "w", **profile) as destination:
        for _, window in dataset.block_windows(1):
            destination.write(dataset.read(1, window=window), 1, window=window)
    return handle.name


def _windows(width: int, height: int, size: int = 1024):
    for top in range(0, height, size):
        for left in range(0, width, size):
            yield rasterio.windows.Window(
                left,
                top,
                min(size, width - left),
                min(size, height - top),
            )


def main() -> None:
    scene_directory = select_scene_directory()
    output_directory = scene_directory / "Output"
    align_enhanced_thermal(
        output_directory / "Enhanced_Thermal" / "Enhanced_Thermal_Image.tif",
        output_directory / "Thermal_TIFF" / "Temperature_Celsius.tif",
        output_directory / "Enhanced_Thermal" / "Enhanced_Thermal_Pix2Pix.tif",
    )


if __name__ == "__main__":
    main()
