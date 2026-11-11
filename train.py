import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Create a small synthetic regression dataset:
# y = 3*x1 - 2*x2 + noise
torch.manual_seed(42)

X = torch.randn(1_000, 2)
y = 3 * X[:, :1] - 2 * X[:, 1:2] + 0.1 * torch.randn(1_000, 1)

loader = DataLoader(
    TensorDataset(X, y),
    batch_size=32,
    shuffle=True,
)

# A feed-forward network with one hidden layer
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train for 10 epochs
model.train()

for epoch in range(200):
    total_loss = 0.0

    for inputs, targets in loader:
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    average_loss = total_loss / len(loader.dataset)
    print(f"Epoch {epoch + 1:2d}/10 | Loss: {average_loss:.4f}")

# Test one prediction
model.eval()

with torch.no_grad():
    sample = torch.tensor([[2.0, 1.0]])
    prediction = model(sample)
    print(f"Prediction: {prediction.item():.3f}")  # Expected near 4