from pathlib import Path

from esrgan_dataset_preparation import enhance_thermal_image


THERMAL_IMAGE = Path(
    r"C:\Users\subhe\OneDrive\Desktop\datasets\scene_017\Output\Thermal_PNG\Thermal_Image.png"
)

CHECKPOINT = Path(
    r"C:\Users\subhe\OneDrive\Desktop\datasets\scene_017\Output\ESRGAN_Dataset\checkpoints\esrgan_epoch_0050.pt"
)

OUTPUT_IMAGE = Path(
    r"C:\Users\subhe\OneDrive\Desktop\datasets\scene_017\Output\Enhanced_Thermal\Enhanced_Thermal_Image.tif"
)


enhanced_image = enhance_thermal_image(
    thermal_path=THERMAL_IMAGE,
    checkpoint_path=CHECKPOINT,
    output_path=OUTPUT_IMAGE,
    tile_size=128,
    device="cuda",
)

print(f"Enhanced image saved at: {enhanced_image}")