import torch
import torch.nn as nn

class AllConvNet(nn.Module):
    def __init__(self):
        super(AllConvNet, self).__init__()

        self.layers = nn.Sequential(
            # 3x3 conv, 16-BN-ReLU x2
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 3x3 conv, 32 stride 2-BN-ReLU
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 3x3 conv, 32-BN-ReLU x2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 3x3 conv, 64 stride 2-BN-ReLU
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 3x3 conv, 64 valid padding-BN-ReLU
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 1x1 conv, 64-BN-ReLU
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Global average pooling
            nn.AdaptiveAvgPool2d(1),

            # Reshape
            nn.Flatten(),

            # Logits for 10 classes (for CIFAR-10)
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

    def get_hilbert_rep()

