import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, nq, T):
        super().__init__()
        self.T, self.nq = T, nq
        self.fc1 = nn.Linear(nq + 3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, (T-1)*nq)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
