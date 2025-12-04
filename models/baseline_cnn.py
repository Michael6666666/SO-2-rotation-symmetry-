"""
Simple 4 layers CNN Baseline model
Target goal：less parameter，fast training，suitable for CPU，compare with Steerable CNN 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBaselineCNN(nn.Module):
    def __init__(self, num_classes=67):
        super(SimpleBaselineCNN, self).__init__()

        # ----- Feature extraction (4 convolution layer) -----
        # Layer 1: input: 3 channel -> Output: 16 channel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Layer 2: 16 -> 32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Layer 3: 32 -> 64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Layer 4: 64 -> 128
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Pooling 
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2) # Prevant overfit

        # ----- Fully connected classification layer -----
        # Input calculation: 128x128 -> pool -> 64x64 -> pool -> 32x32 -> pool -> 16x16 -> pool -> 8x8
        # So final feature map size is 128 channels * 8 * 8
        self.fc_input_dim = 128 * 8 * 8

        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

        print(f"✅ SimpleBaselineCNN initialization finished")
        print(f"   Output classes: {num_classes}")

    def forward(self, x):
        # x: [Batch, 3, 128, 128]

        # Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x: [Batch, 16, 64, 64]

        # Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x: [Batch, 32, 32, 32]

        # Layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # x: [Batch, 64, 16, 16]

        # Layer 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # x: [Batch, 128, 8, 8]

        # Flatten
        x = x.view(-1, self.fc_input_dim)
        x = self.dropout(x)

        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

