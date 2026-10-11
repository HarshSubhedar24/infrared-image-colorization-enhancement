# infrared-image-colorization-enhancement
Infrared Image Colorization and Enhancement for Improved Object Interpretation


                     Landsat 8/9 Satellite Dataset
                                │
                                ▼
                   Scene Selection & Data Acquisition
                                │
                                ▼
                Extract Required Spectral Bands
             ┌────────────────────┴────────────────────┐
             ▼                                         ▼
      RGB Bands (B2,B3,B4)                   Thermal Band (ST_B10)
             │                                         │
             ▼                                         ▼
        RGB Preparation                    Thermal Preprocessing
             │                                         │
             │                                         │
             ▼                                         ▼
      Ground Truth RGB               Processed Thermal Image
             │                                         │
             └────────────────────┬────────────────────┘
                                  ▼
                     ESRGAN Super Resolution
              (Enhancement of Thermal Image Details)
                                  │
                                  ▼
                   Enhanced Thermal Image
                                  │
                                  ▼
                Pix2Pix Image Translation Network
          (Thermal Image → Colorized RGB Image)
                                  │
                                  ▼
                    Generated Colorized RGB Image
                                  │
                                  ▼
               Quantitative & Qualitative Evaluation
          (PSNR, SSIM, LPIPS, FID + Visual Comparison)
                                  │
                                  ▼
                          Final Enhanced Output








Current progression:


RGB preparation
        ↓
Thermal preprocessing
        ↓
ESRGAN dataset preparation
        ↓
ESRGAN training
        ↓
ESRGAN inference
        ↓
Pix2Pix thermal alignment to RGB grid