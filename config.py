"""Configuration for rotation-equivariant CNN project"""

import os
import torch
from pathlib import Path

class Config:
    # Dataset settings
    data_root = './data'  
    img_size = 128
    batch_size = 32
    num_workers = 0

    # Training settings
    num_epochs = 50  
    learning_rate = 0.001
    weight_decay = 1e-4

    # Dataset classes
    dataset_classes = {
        'dtd': 47,
        'kth': 10,
        'cifar10': 10
    }
    mixed_num_classes = 67

    # Augmentation
    rotation_range = 360

    # Paths
    checkpoint_dir = './checkpoints'
    results_dir = './results'

    # Seed
    seed = 42

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        Path(cls.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cls.results_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print configuration summary"""
        print(f"✅ Configuration loaded")
        print(f"📍 Device: {cls.device}")
        print(f"🖼️  Image size: {cls.img_size}x{cls.img_size}")
        print(f"📦 Batch size: {cls.batch_size}")
        print(f"🔄 Epochs: {cls.num_epochs}")
        print(f"\n📊 Classes:")
        for dataset, n in cls.dataset_classes.items():
            print(f"   {dataset.upper()}: {n}")
        print(f"   Mixed: {cls.mixed_num_classes}")