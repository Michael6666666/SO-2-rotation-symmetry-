"""
CIFAR-10 Download and Subsampling Script

Note: CIFAR-10 will be augmented with rotations during training
to help the model learn rotation invariance/equivariance.
"""

import torchvision
from pathlib import Path
import numpy as np
import random
import pickle

def download_cifar10():
    """Download CIFAR-10 dataset using torchvision"""
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)

    print("⏬ Downloading CIFAR-10 dataset...")

    train_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True
    )

    print("✅ CIFAR-10 downloaded!")
    print(f"📊 Training set: {len(train_dataset)} images")
    print(f"📊 Test set: {len(test_dataset)} images")

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"🏷️  Classes: {classes}")

    return train_dataset, test_dataset


def subsample_cifar10(dataset, target_size=5000, seed=42):
    """
    Subsample CIFAR-10 with balanced classes
    
    Args:
        dataset: CIFAR-10 dataset object
        target_size: Total number of samples to keep
        seed: Random seed for reproducibility
    
    Returns:
        List of selected indices
    """
    random.seed(seed)
    np.random.seed(seed)

    n_per_class = target_size // 10

    # Collect indices for each class
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Sample balanced subset
    selected_indices = []
    for cls in range(10):
        indices = class_indices[cls]
        selected = random.sample(indices, min(n_per_class, len(indices)))
        selected_indices.extend(selected)

    print(f"✅ Subsampled: {len(dataset)} → {len(selected_indices)} images")
    
    return selected_indices


def prepare_cifar10_subset():
    """Complete CIFAR-10 preparation: download + subsample + save indices"""
    
    # Download datasets
    train_dataset, test_dataset = download_cifar10()

    # Subsample training set (5000 images)
    print("\n📉 Subsampling training set...")
    train_indices = subsample_cifar10(train_dataset, target_size=5000)
    print(f"   Training: {len(train_indices)} images")

    # Subsample test set (1000 images)
    print("📉 Subsampling test set...")
    test_indices = subsample_cifar10(test_dataset, target_size=1000)
    print(f"   Test: {len(test_indices)} images")

    # Save indices for reproducibility
    indices_path = Path('./data/cifar10_subsampled_indices.pkl')
    with open(indices_path, 'wb') as f:
        pickle.dump({
            'train_indices': train_indices,
            'test_indices': test_indices
        }, f)

    print(f"\n💾 Indices saved to: {indices_path}")
    print("✅ CIFAR-10 preparation complete!")


if __name__ == "__main__":
    prepare_cifar10_subset()