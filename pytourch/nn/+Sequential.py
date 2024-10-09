import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple Sequential model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(5, 10),  # Assuming input data has 5 features
    nn.ReLU(),
    nn.Linear(10, 2),  # Output layer with 2 classes
    nn.Softmax(dim=1)
)

# Define a synthetic dataset and DataLoader (for simplicity)
# In a real-world scenario, you would use your actual dataset and DataLoader
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, input_size):
        self.data = torch.rand((num_samples, input_size))
        self.labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create a dummy dataset and DataLoader
dummy_dataset = DummyDataset(num_samples=100, input_size=5)
dummy_dataloader = torch.utils.data.DataLoader(dummy_dataset, batch_size=10, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, labels in dummy_dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    avg_loss = total_loss / len(dummy_dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluate the trained model on some test data
test_data = torch.rand((5, 5))  # Assuming 5 test samples with 5 features each
test_predictions = model(test_data)
print("\nTest Predictions:")
print(test_predictions)
