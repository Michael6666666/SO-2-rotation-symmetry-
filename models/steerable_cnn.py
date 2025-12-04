"""
Four layers Steerable CNN - using e2cnn package to achieve rotating equivariance
Main feature: Steerable Filters (continuous rotating equivariance)
"""

import torch
import torch.nn as nn
import e2cnn
from e2cnn import gspaces
from e2cnn import nn as e2nn

class SimpleSteerableCNN(nn.Module):
    """
    4 layers Steerable CNN - matching Baseline 4 layers structure
    Using SO(2) group to achieve continuous rotationg equivariance
    """

    def __init__(self, num_classes=67, N=8):
        """
        Args:
            num_classes: Output classes
            N: Rotation level, the more big N，the precise of rotation equivariance has，but more computation
               N=8 representing it can proceed 360°/8 = 45° 's rotation
        """
        super(SimpleSteerableCNN, self).__init__()

        
        self.r2_act = gspaces.Rot2dOnR2(N=N)

        
        in_type = e2nn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])

        # Layer 1: 3 -> 16
        out_type = e2nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.conv1 = e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn1 = e2nn.InnerBatchNorm(out_type)
        self.relu1 = e2nn.ReLU(out_type, inplace=True)
        self.pool1 = e2nn.PointwiseMaxPool(out_type, 2)

        # Layer 2: 16 -> 32 
        in_type = out_type
        out_type = e2nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.conv2 = e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn2 = e2nn.InnerBatchNorm(out_type)
        self.relu2 = e2nn.ReLU(out_type, inplace=True)
        self.pool2 = e2nn.PointwiseMaxPool(out_type, 2)

        # Layer 3: 32 -> 64 
        in_type = out_type
        out_type = e2nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.conv3 = e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn3 = e2nn.InnerBatchNorm(out_type)
        self.relu3 = e2nn.ReLU(out_type, inplace=True)
        self.pool3 = e2nn.PointwiseMaxPool(out_type, 2)

        # Layer 4: 64 -> 128 
        in_type = out_type
        out_type = e2nn.FieldType(self.r2_act, 128*[self.r2_act.regular_repr])
        self.conv4 = e2nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        self.bn4 = e2nn.InnerBatchNorm(out_type)
        self.relu4 = e2nn.ReLU(out_type, inplace=True)
        self.pool4 = e2nn.PointwiseMaxPool(out_type, 2)

        
        self.gpool = e2nn.GroupPooling(out_type)

        
        self.fc_input_dim = 128 * 8 * 8

        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

        print(f"✅ Steerable CNN initialize finished")
        print(f"   Rotation level N: {N}")
        print(f"   Output number of classes: {num_classes}")
        print(f"   Eqivariance: SO(2) continuous rotation")

    def forward(self, x):
        # x: [Batch, 3, 128, 128] - general tensor

        # Transfer to GeometricTensor
        x = e2nn.GeometricTensor(x, self.conv1.in_type)

        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # x: [Batch, 16*N, 64, 64] - GeometricTensor

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # x: [Batch, 32*N, 32, 32]

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        # x: [Batch, 64*N, 16, 16]

        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        # x: [Batch, 128*N, 8, 8]

        # Group Pooling
        x = self.gpool(x)
        # x: [Batch, 128, 8, 8] - Back to general tensor

        # Flatten
        x = x.tensor  # extract general tensor
        x = x.view(x.size(0), -1)

        # Classification
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self):
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


