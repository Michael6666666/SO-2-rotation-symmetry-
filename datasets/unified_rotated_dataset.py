import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import numpy as np
import pickle

class UnifiedRotatedDataset(Dataset):
    """
    Unified Rotated Dataset
    Support: DTD, KTH-TIPS, CIFAR-10
    Function: Random rotated for image, divide train/val/test set 
    """

    def __init__(self,
                 dataset_name='dtd',
                 split='train',
                 img_size=128,
                 rotation_range=360,
                 seed=42,
                 data_root='./data'):
        """
        Args:
            dataset_name: 'dtd', 'kth', or 'cifar10'
            split: 'train', 'val', or 'test'
            img_size
            rotation_range: 0-360 degree
            seed: 
            data_root: 
        """
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.img_size = img_size
        self.rotation_range = rotation_range
        self.data_root = Path(data_root)

        # Setting random seed
        random.seed(seed)
        np.random.seed(seed)

        # Loading data based on dataset name
        if self.dataset_name == 'dtd':
            self._load_dtd()
        elif self.dataset_name == 'kth':
            self._load_kth()
        elif self.dataset_name == 'cifar10':
            self._load_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Define data augmentation
        self._setup_transforms()

        print(f"✅ {self.dataset_name.upper()} {split} finish loading: {len(self.samples)} images")

    def _load_dtd(self):
        """Load DTD dataset"""
        dtd_path = self.data_root / 'dtd' / 'images'

        if not dtd_path.exists():
            raise FileNotFoundError(f"DTD dataset path not exist: {dtd_path}")

        # Find all Classes
        self.classes = sorted([d.name for d in dtd_path.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all images
        self.samples = []
        for cls in self.classes:
            cls_path = dtd_path / cls
            for img_path in cls_path.glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[cls]))

        # Shuffle and divide to test, val and training set
        random.shuffle(self.samples)
        self._split_dataset(train_ratio=0.7, val_ratio=0.15)

    def _load_kth(self):
        """Load KTH-TIPS dataset"""
        kth_path = self.data_root / 'kth_tips' / 'KTH_TIPS'

        if not kth_path.exists():
            raise FileNotFoundError(f"KTH-TIPS path not exist: {kth_path}")

        # Find all Classes
        self.classes = sorted([d.name for d in kth_path.iterdir()
                              if d.is_dir() and not d.name.startswith('.')])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all images
        self.samples = []
        for cls in self.classes:
            cls_path = kth_path / cls
            for img_path in list(cls_path.glob('*.png')) + list(cls_path.glob('*.jpg')):
                self.samples.append((str(img_path), self.class_to_idx[cls]))

        # Shuffle and divide to test, val and training set
        random.shuffle(self.samples)
        self._split_dataset(train_ratio=0.7, val_ratio=0.15)

    def _load_cifar10(self):
        """Load CIFAR-10 dataset which after subsampled"""
        import torchvision

        # Load CIFAR-10 dataset
        is_train = (self.split == 'train')
        cifar_dataset = torchvision.datasets.CIFAR10(
            root=str(self.data_root),
            train=is_train,
            download=False
        )

        # CIFAR-10 classes
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Load subsampled indices
        indices_path = self.data_root / 'cifar10_subsampled_indices.pkl'
        if indices_path.exists():
            with open(indices_path, 'rb') as f:
                indices_dict = pickle.load(f)

            if is_train:
                indices = indices_dict['train_indices']
            else:
                indices = indices_dict['test_indices']

            # Using subsampled indices
            self.samples = [(idx, cifar_dataset[idx][1]) for idx in indices]
        else:
            # If not subsampled，using all the data in Cifar 10
            print("⚠️  Did not find subsampled indices，using all CIFAR-10 data")
            self.samples = [(idx, label) for idx, (_, label) in enumerate(cifar_dataset)]

        
        self.cifar_dataset = cifar_dataset

        # Split training, val and test datasets for Cifar 10
        if is_train:
            # Training set split to train and val dataset
            n_total = len(self.samples)
            n_train = int(0.85 * n_total)  # 85% train, 15% val

            if self.split == 'train':
                self.samples = self.samples[:n_train]
            else:  # val
                self.samples = self.samples[n_train:]

    def _split_dataset(self, train_ratio=0.7, val_ratio=0.15):
        
        n_total = len(self.samples)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        if self.split == 'train':
            self.samples = self.samples[:n_train]
        elif self.split == 'val':
            self.samples = self.samples[n_train:n_train+n_val]
        else:  # test
            self.samples = self.samples[n_train+n_val:]

    def _setup_transforms(self):
        """Setting data augmentation"""
        if self.split == 'train':
            # Training set data augmentation：Relatively strong augmentation
            self.transform = transforms.Compose([
                transforms.Resize(int(self.img_size * 1.1)),
                transforms.RandomCrop(self.img_size),
                transforms.RandomRotation(self.rotation_range),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # Val/Test：Relatively weak augmentation
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size),
                transforms.RandomRotation(self.rotation_range),  
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.dataset_name == 'cifar10':
            
            cifar_idx, label = self.samples[idx]
            img, _ = self.cifar_dataset[cifar_idx]
            img = self.transform(img)
        else:
            
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)

        return img, label

    def get_num_classes(self):
        
        return len(self.classes)


# ==================== Testing code ====================
if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing unified dataset loader")
    print("="*70)

    # Testing 3 datasets
    datasets_to_test = ['dtd', 'kth', 'cifar10']

    for dataset_name in datasets_to_test:
        print(f"\n{'='*70}")
        print(f"📊 Testing {dataset_name.upper()}")
        print(f"{'='*70}")

        try:
            
            train_dataset = UnifiedRotatedDataset(
                dataset_name=dataset_name,
                split='train',
                img_size=128,
                rotation_range=360
            )

            val_dataset = UnifiedRotatedDataset(
                dataset_name=dataset_name,
                split='val',
                img_size=128,
                rotation_range=360
            )

            test_dataset = UnifiedRotatedDataset(
                dataset_name=dataset_name,
                split='test',
                img_size=128,
                rotation_range=360
            )

            
            print(f"  Number of classes: {train_dataset.get_num_classes()}")
            print(f"  Classes name: {train_dataset.classes[:3]}...")

        
            img, label = train_dataset[0]
            print(f"  Image shape: {img.shape}")
            print(f"  Label range: 0-{train_dataset.get_num_classes()-1}")

            print(f"✅ {dataset_name.upper()} pass test！")

        except Exception as e:
            print(f"❌ {dataset_name.upper()} fail test: {e}")

    print("\n" + "="*70)
    print("🎉 Finish unified dataset loader test ！")
    print("="*70)