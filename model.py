import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet-like architecture adapted for small grayscale images (e.g. 28x28).
    - input_channels: number of image channels (1 for ChestMNIST)
    - num_classes: number of output classes
    """
    def __init__(self, input_channels=1, num_classes=2):
        super(AlexNet, self).__init__()
        # Feature extractor (AlexNet-inspired but adjusted for small images)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28 -> 14

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14 -> 7

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 7 -> 3
        )

        # Adaptive pooling kept small (features produce 256 x 3 x 3 for 28x28 input)
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        # Classifier (reduced sizes compared to original AlexNet)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    print("--- Menguji Model 'AlexNet (adapted)' ---")

    NUM_CLASSES = 2
    IN_CHANNELS = 1
    model = AlexNet(input_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Arsitektur Model:")
    print(model)

    # Test dengan dummy input (batch, channels, height, width)
    batch_size = 8
    dummy_input = torch.randn(batch_size, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    print("\nOutput shape:", output.shape)