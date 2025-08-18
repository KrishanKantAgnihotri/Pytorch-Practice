import torch
import torch.nn as nn
import torch.optim as optim

# 🔧 Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Swish activation

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# 🧱 Backbone Network
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ConvBlock(3, 32, 3, 1, 1)
        self.layer2 = ConvBlock(32, 64, 3, 2, 1)
        self.layer3 = ConvBlock(64, 128, 3, 2, 1)
        self.layer4 = ConvBlock(128, 256, 3, 2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

# 🎯 Detection Head
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, (num_classes + 5) * 3, 1)

    def forward(self, x):
        return self.conv(x)

# 🧩 Full Model
class YOLOv5Lite(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.head = DetectionHead(256, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# 🧮 Simplified YOLO Loss
class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        # Assumes targets are aligned with predictions
        obj_pred = predictions[:, 4::(targets.shape[1] // 3), :, :]
        obj_target = targets[:, 4::(targets.shape[1] // 3), :, :]

        box_pred = predictions[:, 0:4, :, :]
        box_target = targets[:, 0:4, :, :]

        class_pred = predictions[:, 5:, :, :]
        class_target = targets[:, 5:, :, :]

        obj_loss = self.bce(obj_pred, obj_target)
        box_loss = self.mse(box_pred, box_target)
        class_loss = self.bce(class_pred, class_target)

        return obj_loss + box_loss + class_loss

# 🏋️ Training Loop (Simplified)
def train(model, dataloader, loss_fn, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for images, targets in dataloader:
            predictions = model(images)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 🧪 Example Usage
if __name__ == "__main__":
    num_classes = 20
    model = YOLOv5Lite(num_classes)
    loss_fn = YOLOLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Dummy data for testing
    dummy_images = torch.randn(2, 3, 640, 640)
    dummy_targets = torch.randn(2, (num_classes + 5) * 3, 80, 80)  # Assuming 8x downsampling

    # Dummy dataloader
    dataloader = [(dummy_images, dummy_targets)] * 5

    train(model, dataloader, loss_fn, optimizer, epochs=3)