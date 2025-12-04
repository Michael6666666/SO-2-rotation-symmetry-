"""Mixed dataset combining DTD, KTH-TIPS, and CIFAR-10"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets.unified_rotated_dataset import UnifiedRotatedDataset


class MixedDataset(Dataset):
    """Combine three datasets with label offset"""

    def __init__(self, split='train', img_size=128, rotation_range=360):
        self.split = split

        print(f"🔄 Loading {split} dataset's three subsets...")

        self.dtd = UnifiedRotatedDataset(dataset_name='dtd', split=split,
                                       img_size=img_size, rotation_range=rotation_range)

        self.kth = UnifiedRotatedDataset(dataset_name='kth', split=split,
                                       img_size=img_size, rotation_range=rotation_range)

        self.cifar = UnifiedRotatedDataset(dataset_name='cifar10', split=split,
                                         img_size=img_size, rotation_range=rotation_range)

        # Label offsets
        self.dtd_offset = 0
        self.kth_offset = 47  
        self.cifar_offset = 57 

        # Dataset lengths
        self.len_dtd = len(self.dtd)
        self.len_kth = len(self.kth)
        self.len_cifar = len(self.cifar)
        self.total_len = self.len_dtd + self.len_kth + self.len_cifar

        print(f"✅ Mixed finished ({split}): Total {self.total_len} images")
        print(f"   Structure: DTD({self.len_dtd}) + KTH({self.len_kth}) + CIFAR10({self.len_cifar})")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < self.len_dtd:
            img, label = self.dtd[idx]
            final_label = label + self.dtd_offset
        elif idx < (self.len_dtd + self.len_kth):
            real_idx = idx - self.len_dtd
            img, label = self.kth[real_idx]
            final_label = label + self.kth_offset
        else:
            real_idx = idx - (self.len_dtd + self.len_kth)
            img, label = self.cifar[real_idx]
            final_label = label + self.cifar_offset

        return img, final_label


def get_dataloaders(config):
    """Create train/val/test DataLoaders for mixed dataset"""
    
    print(f"🔄 Creating DataLoaders...")

    train_dataset = MixedDataset(
        split='train',
        img_size=config.img_size,
        rotation_range=config.rotation_range
    )

    val_dataset = MixedDataset(
        split='val',
        img_size=config.img_size,
        rotation_range=config.rotation_range
    )

    test_dataset = MixedDataset(
        split='test',
        img_size=config.img_size,
        rotation_range=config.rotation_range
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )

    print(f"\n✅ Datasets loading finished！")
    print(f"📊 Training set: {len(train_dataset)} images")
    print(f"📊 Validation set: {len(val_dataset)} images")
    print(f"📊 Testing set: {len(test_dataset)} images")
    print(f"🏷️  Total classes: {config.mixed_num_classes}")
    print(f"   (DTD: {train_dataset.len_dtd} + KTH: {train_dataset.len_kth} + CIFAR: {train_dataset.len_cifar})")

    return train_loader, val_loader, test_loader

