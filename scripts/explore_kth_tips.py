import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

def visualize_kth_tips():
    """Visulize KTH-TIPS textual sample"""

    
    kth_path = Path('./data/kth/KTH_TIPS')

    if not kth_path.exists():
        print("❌ DId not find KTH-TIPS datasets")
        print("💡 Path should be: ./data/kth_tips/KTH_TIPS/")
        return

    
    classes = sorted([d.name for d in kth_path.iterdir()
                     if d.is_dir() and not d.name.startswith('.')])

    if len(classes) == 0:
        print("❌ Did not find the KTH-TIPS path")
        return

    print(f"✅ Find {len(classes)} classes")

    # Showing all classes in the dataset
    n_classes = min(len(classes), 10)
    selected_classes = classes[:n_classes]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('KTH-TIPS Texture Dataset Samples', fontsize=16)

    for idx, cls in enumerate(selected_classes):
        row = idx // 5
        col = idx % 5

        # Choosing one image for each classes
        cls_path = kth_path / cls
        images = list(cls_path.glob('*.png')) + list(cls_path.glob('*.jpg'))

        if len(images) == 0:
            print(f"⚠️  {cls}: did not find images")
            continue

        sample = random.choice(images)

        img = Image.open(sample)
        ax = axes[row, col]
        ax.imshow(img, cmap='gray') # They are gray images 
        
        display_name = cls.replace('_', ' ').title()
        ax.set_title(display_name, fontsize=10)
        ax.axis('off')

    
    for idx in range(n_classes, 10):
        row = idx // 5
        col = idx % 5
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('kth_tips_samples.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved kth_tips_samples.png")

    # Print out KTH-TIPs dataset statistics
    print(f"\n📊 KTH-TIPS dataset statisitcs:")
    print("="*60)
    total_images = 0
    for cls in classes:
        cls_path = kth_path / cls
        n_images = len(list(cls_path.glob('*.png'))) + len(list(cls_path.glob('*.jpg')))
        total_images += n_images
        display_name = cls.replace('_', ' ').title()
        print(f"  {display_name:<25} {n_images:>3} images")

    print("="*60)
    print(f"  Total classes: {len(classes)}")
    print(f"  Total images: {total_images}")
    print(f"  Average images number for each class: {total_images // len(classes)} iamges")

if __name__ == "__main__":
    visualize_kth_tips()