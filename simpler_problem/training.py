import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from model import Net


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        inputs = self.data[idx][0]
        labels = torch.flatten(self.data[idx][1])
        return inputs, labels

if __name__ == "__main__":

    data = torch.load('trajectories_3000.pt', weights_only=True)

    # Create dataset
    dataset = CustomDataset(data)
    

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    taset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    net = Net(7, 15)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001)

    N_epoch = 500
    print_every = 24

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
    torch.save(net.state_dict(), 'trained_model_box_test.pth')
