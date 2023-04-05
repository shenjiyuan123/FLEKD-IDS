import torch.nn as nn
import torch.nn.functional as F
import torch

class MyModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(100, 100)
        self.out = nn.Linear(100, output_dim)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = F.relu(self.layer4(out))
        out = self.out(out)
        return out

    def predict(self, x):
        _, predicted = torch.max(F.softmax(self.forward(x), dim=1), 1)
        return predicted
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, output_dim)

    def forward(self, x):
        out = F.relu(self.fc_1(x))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out
    
    def predict(self, x):
        _, predicted = torch.max(F.softmax(self.forward(x), dim=1), 1)
        return predicted