import os
import urllib.request
import tarfile
from pathlib import Path
import shutil

def download_kth_tips(data_dir=Path('./data')):
    """Download and unzip KTH-TIPS datasets"""

    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    kth_dir = data_dir / 'kth_tips'

    # Checking if the dataset already exist
    if kth_dir.exists() and len(list(kth_dir.glob('*'))) > 0:
        print("✅ KTH-TIPS dataset already exist")
        return

    kth_dir.mkdir(exist_ok=True)

    # KTH-TIPS dataset's url
    url = "https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_grey_200x200.tar"
    tar_path = data_dir / "kth_tips.tar"

    print("⏬ Downloading KTH-TIPS dataset...")
    print(f"   URL: {url}")

    try:
        urllib.request.urlretrieve(url, tar_path)
        print("✅ Finish downloading KTH-TIPS datasets！")
    except Exception as e:
        print(f"❌ Failed to download the KTH-TIPS datasets: {e}")
        # print("💡 Using other method ...")
        # If download file, provide a way to download the dataset
        print("\n Please download KTH-TIPS dataset manually：")
        print("1. Check the website: https://www.csc.kth.se/cvap/databases/kth-tips/")
        print("2. Download kth_tips_grey_200x200.tar")
        print("3. Put under ./data/ file path")
        return

    print("📦 Unzipping...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(kth_dir)
    print("✅ Finish unzipping！")

    # Clean cache
    tar_path.unlink()
    print("🗑️  Clean cache")

    # Check KTH-TIPS dataset information
    print("\n📊 Dataset information:")

    # KTH-TIPS has 10 classes
    classes = sorted([d.name for d in kth_dir.iterdir() if d.is_dir()])
    print(f"  - Number of Classes: {len(classes)}")
    print(f"  - Classes: {classes}")

    # Counting total number of images in KTH-TIPS
    total_images = 0
    for cls in classes:
        cls_path = kth_dir / cls
        n_images = len(list(cls_path.glob('*.png'))) + len(list(cls_path.glob('*.jpg')))
        total_images += n_images

    print(f"  - Total image number in KTH-TIPS dataset: {total_images}")

if __name__ == "__main__":
    download_kth_tips()