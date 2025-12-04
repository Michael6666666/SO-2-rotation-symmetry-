import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import torchvision
import numpy as np

def visualize_all_datasets_final():
    """Visualize three datasets to do some comparision"""

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(3, 6, hspace=0.3, wspace=0.2)

    fig.suptitle('Three Datasets Comparison', fontsize=18, fontweight='bold')

    # ===== Row 1: DTD =====
    print("📊 Loading DTD...")
    dtd_path = Path('./data/dtd/images')
    dtd_classes = sorted([d.name for d in dtd_path.iterdir() if d.is_dir()])
    selected_dtd = random.sample(dtd_classes, 6)

    for col, cls in enumerate(selected_dtd):
        ax = fig.add_subplot(gs[0, col])
        cls_path = dtd_path / cls
        images = list(cls_path.glob('*.jpg'))
        if images:
            img = Image.open(random.choice(images))
            ax.imshow(img)
            ax.set_title(f'{cls}', fontsize=9)
            ax.axis('off')

    fig.text(0.02, 0.83, 'DTD\n(Textures)\nColor\nHigh-res',
             fontsize=11, fontweight='bold', va='center', ha='left')

    # ===== Row 2: KTH-TIPS =====
    print("📊 Loading KTH-TIPS...")
    
    kth_base = Path('./data/kth/KTH_TIPS')

    if not kth_base.exists():
        print("❌ KTH_TIPS path not exist！")
        kth_classes = []
    else:
        kth_classes = sorted([d.name for d in kth_base.iterdir()
                             if d.is_dir() and not d.name.startswith('.')])
        print(f"✅ Found {len(kth_classes)} number of KTH class")
        print(f"   Classes: {kth_classes[:3]}...")

    for col in range(min(6, len(kth_classes))):
        ax = fig.add_subplot(gs[1, col])
        cls = kth_classes[col]
        cls_path = kth_base / cls
        images = list(cls_path.glob('*.png'))

        if images:
            img = Image.open(random.choice(images))
            ax.imshow(img, cmap='gray')
            
            display_name = cls.replace('_', ' ').title()
            ax.set_title(f'{display_name}', fontsize=9)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No image', ha='center', va='center')
            ax.axis('off')

    fig.text(0.02, 0.50, 'KTH-TIPS\n(Textures)\nGrayscale\n200×200',
             fontsize=11, fontweight='bold', va='center', ha='left')

    # ===== Row 3: CIFAR-10 =====
    print("📊 Loading CIFAR-10...")
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog']
    data_dir = Path('./data')
    dataset = torchvision.datasets.CIFAR10(root=str(data_dir), train=True, download=False)

    for col, target_cls in enumerate(range(6)):
        ax = fig.add_subplot(gs[2, col])
        for img, label in dataset:
            if label == target_cls:
                img_array = np.array(img)
                ax.imshow(img_array, interpolation='nearest')
                ax.set_title(f'{cifar_classes[col].title()}', fontsize=9)
                ax.axis('off')
                break

    fig.text(0.02, 0.17, 'CIFAR-10\n(Objects)\nColor\n32×32',
             fontsize=11, fontweight='bold', va='center', ha='left')

    plt.savefig('all_datasets_comparison_final.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved all_datasets_comparison_final.png")

    # Print out comparision information
    print("\n" + "="*70)
    print("📊 Three datasets statistics comparison")
    print("="*70)
    print(f"{'Dataset':<15} {'Type':<20} {'Resolution':<15} {'Classes':<10}")
    print("-"*70)
    print(f"{'DTD':<15} {'Color Textures':<20} {'Variable (high)':<15} {len(dtd_classes):<10}")
    print(f"{'KTH-TIPS':<15} {'Gray Textures':<20} {'200×200':<15} {len(kth_classes):<10}")
    print(f"{'CIFAR-10':<15} {'Natural Objects':<20} {'32×32 (low)':<15} {'10':<10}")
    print("="*70)

    # Count images for each dataset
    dtd_count = sum(len(list((dtd_path/c).glob('*.jpg'))) for c in dtd_classes)
    kth_count = sum(len(list((kth_base/c).glob('*.png'))) for c in kth_classes)

    print(f"\n📷 Number of image:")
    print(f"  DTD:      {dtd_count:>6} images")
    print(f"  KTH-TIPS: {kth_count:>6} images")
    print(f"  CIFAR-10 after subsampled: {5000:>6} images")
    print(f"  Total:     {dtd_count + kth_count + 5000:>6} images")

    print("\n💡 Dataset characteristic:")
    print("  • DTD: High-resolution color textures with diverse categories")
    print("  • KTH-TIPS: Gray-scale texture, varying illumination or scale")
    print("  • CIFAR-10: Low-resolution natural objects, classic benchmark dataset")

if __name__ == "__main__":
    visualize_all_datasets_final()