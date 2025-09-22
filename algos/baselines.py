import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.networks import MLP


class BaselineMLP(MLP):
    pass


class Baseline_MNISTClassifier(nn.Module):
    def __init__(self, keysize=32):
        super(Baseline_MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)

        self.fc1 = nn.Linear(512 + keysize, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

        self.bn2 = nn.BatchNorm2d(32, 0.8)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.bn4 = nn.BatchNorm2d(128, 0.8)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, k: torch.Tensor):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.bn2(self.dropout(self.relu(self.conv2(x))))
        x = self.bn3(self.dropout(self.relu(self.conv3(x))))
        x = self.bn4(self.dropout(self.relu(self.conv4(x))))
        x = x.reshape(x.shape[0], -1)
        i = torch.concat((x, k), dim=1)
        x = self.relu(self.fc1(i))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

    def sample(self, x: torch.Tensor, k: torch.Tensor):
        x = self.forward(x, k)
        x = torch.multinomial(x / torch.sum(x), 1)
        return x
