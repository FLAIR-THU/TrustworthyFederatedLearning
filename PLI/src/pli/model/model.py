import torch
import torch.nn as nn
import torch.nn.functional as F


class LM(nn.Module):
    def __init__(self, input_dim=64 * 64, output_dim=20, hidden_dim=None, channel=1):
        super(LM, self).__init__()
        self.fla = nn.Flatten()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fla(x)
        x = self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim=64 * 64, hidden_dim=100, output_dim=20):
        super(MLP, self).__init__()
        self.fla = nn.Flatten()
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fla(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


class OneCNN(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=12800, output_dim=20, channel=1):
        super(OneCNN, self).__init__()
        self.conv1 = nn.Conv2d(channel, 32, (3, 3))
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = torch.relu(x)
        x = F.max_pool2d(x, (3, 3))
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        return x


def get_model_class(model_type):
    if model_type == "LM":
        model_class = LM
    elif model_type == "MLP":
        model_class = MLP
    elif model_type == "CNN":
        model_class = OneCNN
    return model_class
