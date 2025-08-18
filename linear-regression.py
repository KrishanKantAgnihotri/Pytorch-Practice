import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------
# 1. Fake Data (y = 2x + 3)
# ------------------------------
x = torch.linspace(-5, 5, 100).unsqueeze(1)   # shape: (100,1)
y = 2 * x + 3 + 0.2 * torch.randn(x.size())   # add small noise

# ------------------------------
# 2. Model
# ------------------------------
model = nn.Linear(1, 1)   # input=1 feature, output=1 target

# ------------------------------
# 3. Loss + Optimizer
# ------------------------------
criterion = nn.MSELoss()       # mean squared error
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ------------------------------
# 4. Training Loop
# ------------------------------
for epoch in range(100):
    # Forward
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ------------------------------
# 5. Inference
# ------------------------------
test_val = torch.tensor([[4.0]])  # input x=4
pred = model(test_val)
print(f"Prediction for x=4: {pred.item():.2f}")
