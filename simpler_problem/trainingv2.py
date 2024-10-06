import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        return x, y

# Simple MLP model
class MLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=98):  # Output is 14*7=98
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


data = torch.load('trajectories_3000.pt', weights_only=True)

# Dataset and DataLoader
dataset = CustomDataset(data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = MLP(input_size=10, hidden_size=256, output_size=98)
criterion = nn.MSELoss()  # Assuming regression for now
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training Loop
def train(model, dataloader, criterion, optimizer, epochs=1000):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}')
    return model

# Evaluation Loop
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    print(f'Evaluation Loss: {total_loss/len(dataloader):.4f}')
if __name__ == "__main__":
    # Run training
    model = train(model, dataloader, criterion, optimizer, epochs=4000)

    # Run evaluatio
    # Save the trained model
    torch.save(model.state_dict(), 'trained_model_box_test.pth')