import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

def visualize_dtd():
    """Visualize the DTD dataset textual sample"""

    dtd_path = Path('./data/dtd/images')
    classes = sorted([d.name for d in dtd_path.iterdir() if d.is_dir()])

    # Randomly choose 6 class sample
    selected_classes = random.sample(classes, 6)

    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    fig.suptitle('DTD Texture Dataset Samples', fontsize=16)

    for col, cls in enumerate(selected_classes):
        # Choosing two pictures from each classes
        cls_path = dtd_path / cls
        images = list(cls_path.glob('*.jpg'))
        samples = random.sample(images, 2)

        for row, img_path in enumerate(samples):
            img = Image.open(img_path)
            ax = axes[row, col]
            ax.imshow(img)
            if row == 0:
                ax.set_title(cls, fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('dtd_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Saved dtd_samples.png")

    # Printing out dataset statistics
    print(f"\n📊 Dataset Statistics:")
    total_images = 0
    for cls in classes:
        cls_path = dtd_path / cls
        n_images = len(list(cls_path.glob('*.jpg')))
        total_images += n_images

    print(f"  - Total number of classes: {len(classes)}")
    print(f"  - Total images: {total_images}")
    print(f"  - Average images number for each class: {total_images // len(classes)} images")

if __name__ == "__main__":
    visualize_dtd()