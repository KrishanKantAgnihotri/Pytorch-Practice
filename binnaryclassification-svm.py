import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 🔄 Transform: Convert images to tensors and flatten them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
])

# 📥 Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000)

# 🧠 Define Multiclass Model
class MultiClassModel(nn.Module):
    def __init__(self):
        super(MultiClassModel, self).__init__()
        self.linear = nn.Linear(784, 10)  # 10 output classes

    def forward(self, x):
        return self.linear(x)  # No softmax here; CrossEntropyLoss handles it

model = MultiClassModel()
criterion = nn.CrossEntropyLoss()  # For multiclass classification
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 🔁 Training Loop
for epoch in range(5):  # You can increase this for better accuracy
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)  # Shape: [batch_size, 10]
        loss = criterion(outputs, labels)  # labels: [batch_size]
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 🔍 Prediction on test set
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get class with highest score
        print("Sample predictions:", predicted[:10])
        break