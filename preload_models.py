"""
Setup PyTorch model cache from local files + download missing models
This prevents slow downloads during runtime/API requests
"""
import os
import shutil
from pathlib import Path
from torchvision import models

print("=" * 70)
print("Setting up PyTorch model cache...")
print("=" * 70)

# Setup cache directory
cache_dir = os.getenv('TORCH_HOME', '/root/.cache/torch')
checkpoint_dir = Path(cache_dir) / 'hub' / 'checkpoints'
checkpoint_dir.mkdir(parents=True, exist_ok=True)

models_setup = []

# Local models folder
local_models = Path('/app/models')

# 1. Copy mobilenet from local models folder (if exists)
print("\n[1/3] Setting up mobilenet_v3_large...")
mobilenet_file = 'mobilenet_v3_large-8738ca79.pth'
local_mobilenet = local_models / mobilenet_file
cached_mobilenet = checkpoint_dir / mobilenet_file

if local_mobilenet.exists():
    print(f"   Copying from local: {local_mobilenet}")
    shutil.copy2(local_mobilenet, cached_mobilenet)
    models_setup.append(f"mobilenet_v3_large (copied from local)")
    print("✓ mobilenet_v3_large cached from local file")
else:
    print("   Local file not found, will download on first use")

# 2. Download or copy RegNet X 800MF
print("\n[2/3] Setting up regnet_x_800mf (IMAGENET1K_V2)...")
regnet_file = 'regnet_x_800mf-94a99ebd.pth'
local_regnet = local_models / regnet_file
cached_regnet = checkpoint_dir / regnet_file

if local_regnet.exists():
    print(f"   Copying from local: {local_regnet}")
    shutil.copy2(local_regnet, cached_regnet)
    models_setup.append(f"regnet_x_800mf (copied from local)")
    print("✓ regnet_x_800mf cached from local file")
else:
    print("   Downloading regnet_x_800mf...")
    try:
        models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2)
        models_setup.append("regnet_x_800mf (downloaded)")
        print("✓ regnet_x_800mf downloaded")
    except Exception as e:
        print(f"   Warning: Could not download regnet: {e}")

# 3. Setup LRASPP MobileNetV3 segmentation model
print("\n[3/3] Setting up lraspp_mobilenet_v3_large (segmentation)...")
try:
    models.segmentation.lraspp_mobilenet_v3_large(pretrained=True)
    models_setup.append("lraspp_mobilenet_v3_large (loaded)")
    print("✓ lraspp_mobilenet_v3_large loaded")
except Exception as e:
    print(f"   Warning: Could not load segmentation model: {e}")

print("\n" + "=" * 70)
print("✅ PyTorch model cache setup complete!")
print("\nModels setup:")
for i, model_name in enumerate(models_setup, 1):
    print(f"  {i}. {model_name}")

cache_dir = os.getenv('TORCH_HOME', '/root/.cache/torch')
print(f"\nCache location: {cache_dir}/hub/checkpoints/")

# List cached files
cached_files = list(checkpoint_dir.glob('*.pth'))
if cached_files:
    print(f"\nCached model files ({len(cached_files)}):")
    for f in cached_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
print("=" * 70)
