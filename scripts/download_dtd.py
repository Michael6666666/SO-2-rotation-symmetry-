import os
import urllib.request
import tarfile
from pathlib import Path

def download_dtd(data_dir=Path('./data')):
    """Download and unzip the DTD dataset"""

    # Create data path
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    # DTD dataset file url
    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    tar_path = data_dir / "dtd.tar.gz"

    # Checking if the DTD dataset exists
    if (data_dir / "dtd").exists():
        print("✅ DTD dataset already exist")
        return

    print("⏬ Downloading DTD datasets...")
    urllib.request.urlretrieve(url, tar_path)
    print("✅ Finish downloading DTD datasets！")

    print("📦 Unziped the DTD datasets...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(data_dir)
    print("✅ Finish unziping the datasets！")

    # Delete cache
    tar_path.unlink()
    print("🗑️  Clean the cache")

    # Looking dataset information
    dtd_path = data_dir / "dtd" / "images"
    classes = sorted([d.name for d in dtd_path.iterdir() if d.is_dir()])
    print(f"\n📊 Dataset information:")
    print(f"  - Number of Classes: {len(classes)}")
    print(f"  - First 5 classes: {classes[:5]}")

if __name__ == "__main__":
    download_dtd()