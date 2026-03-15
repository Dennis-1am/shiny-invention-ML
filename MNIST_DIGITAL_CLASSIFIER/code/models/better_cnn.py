import torch.nn as nn


class BetterCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input shape: (Batch, 1, 28, 28)
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),     # Out: (32, 28, 28)
            nn.BatchNorm2d(32),                 # Normalizes the 32 features
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),    # Out: (64, 28, 28)
            nn.BatchNorm2d(64),                 # Normalizes the 64 features
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # Out: (64, 14, 14)
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128), # 64 channels * 14 * 14 image size
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.main(x)
