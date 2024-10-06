import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


# Simple MLP model
class MLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=2048, output_size=98):  # Output is 14*7=98
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x



class TorchDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx][0]
        traj = self.data[idx][1]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(traj, dtype=torch.float32)

if __name__ == "__main__":

    data = torch.load('data_3000.pt',  weights_only=True)
    T = 14
    nq = 7

    # Create dataset
    dataset = TorchDataset(data)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    taset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    net = MLP(input_size=14,output_size= T * nq)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    N_epoch = 250
    print_every = 120

    print(f"Training data size = {len(train_loader.dataset)}")
    print(f"Validation data size = {len(val_loader.dataset)}")

    for epoch in range(N_epoch):
        running_loss = 0.0
        net.train()  # Set the network to training mode
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_every == print_every-1:  # print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{N_epoch}], Step [{i + 1}/{len(train_loader)}], Training Loss: {running_loss / print_every:.4f}')
                running_loss = 0.0

        # Evaluation step
        net.eval()  # Set the network to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{N_epoch}], Validation Loss: {val_loss:.4f}')

    print('Finished Training')

    # Save the trained model
    torch.save(net.state_dict(), 'trained_model_box_.pth')
