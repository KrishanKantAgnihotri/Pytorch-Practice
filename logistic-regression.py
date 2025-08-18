import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ------------------------------
# 1. Dataset
# ------------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ------------------------------
# 2. Model (Simple Linear)
# ------------------------------
class SimpleLinear(nn.Module):
    def __init__(self):
        super(SimpleLinear, self).__init__()
        self.fc = nn.Linear(28*28, 10)  # flatten image -> 10 classes

    def forward(self, x):
        x = x.view(-1, 28*28)  # flatten image
        return self.fc(x)

model = SimpleLinear()

# ------------------------------
# 3. Training Setup
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ------------------------------
# 4. Training Loop
# ------------------------------
for epoch in range(2):  # just 2 epochs for speed
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ------------------------------
# 5. Inference / Accuracy
# ------------------------------
correct, total = 0, 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"✅ Accuracy: {100 * correct / total:.2f}%")
