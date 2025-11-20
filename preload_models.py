"""
Pre-download PyTorch models during Docker build to avoid runtime downloads
"""
import torch
from torchvision import models

print("=" * 60)
print("Preloading PyTorch models to cache...")
print("=" * 60)

try:
    # Download regnet model
    print("\n[1/2] Downloading regnet_x_800mf...")
    try:
        # Try new API first (torchvision >= 0.13)
        models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.DEFAULT)
    except:
        # Fallback to old API
        models.regnet_x_800mf(pretrained=True)
    print("✓ regnet_x_800mf downloaded")

    # Download mobilenet model
    print("\n[2/2] Downloading mobilenet_v3_large...")
    try:
        # Try new API first
        models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    except:
        # Fallback to old API
        models.mobilenet_v3_large(pretrained=True)
    print("✓ mobilenet_v3_large downloaded")

    print("\n" + "=" * 60)
    print("All PyTorch models preloaded successfully!")
    print("Models are cached in: /root/.cache/torch/hub/checkpoints/")
    print("=" * 60)

except Exception as e:
    print(f"\n⚠ Warning: Failed to preload models: {e}")
    print("Models will be downloaded on first request instead.")
    exit(0)  # Don't fail the build
