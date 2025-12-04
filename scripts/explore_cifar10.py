import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import pickle

def visualize_cifar10():
    """Visualize CIFAR-10 dataset sample"""

    import torchvision

    # Loading CIFAR-10 dataset
    data_dir = Path('./data')
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=False
    )

    # CIFAR-10 dataset Classes name
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Choosing 2 images in each class
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    fig.suptitle('CIFAR-10 Dataset Samples (Before & After Rotation)', fontsize=16)

    # Find samples for each class
    class_samples = {i: [] for i in range(10)}
    for idx, (img, label) in enumerate(dataset):
        if len(class_samples[label]) < 2:
            class_samples[label].append((img, idx))
        if all(len(samples) >= 2 for samples in class_samples.values()):
            break

    for col, cls_idx in enumerate(range(10)):
        # Orginal image
        img, idx = class_samples[cls_idx][0]
        ax = axes[0, col]
        ax.imshow(img)
        ax.set_title(classes[cls_idx], fontsize=9)
        ax.axis('off')

        # Images after rotated 90 degree
        img_rotated = img.rotate(90)
        ax = axes[1, col]
        ax.imshow(img_rotated)
        ax.set_title('Rotated 90°', fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Saved cifar10_samples.png")

    # Print out Cifar10 dataset statistics
    print(f"\n📊 CIFAR-10 Statistics:")
    print(f"  - Total classes: {len(classes)}")
    print(f"  - Classes: {classes}")
    print(f"  - Orginal images number: {len(dataset)} images")

    # 
    indices_path = Path('./data/cifar10_subsampled_indices.pkl')
    if indices_path.exists():
        with open(indices_path, 'rb') as f:
            indices = pickle.load(f)
        print(f"  - Subsampled training set: {len(indices['train_indices'])} images")
        print(f"  - Subsampled testing set: {len(indices['test_indices'])} images")

if __name__ == "__main__":
    visualize_cifar10()