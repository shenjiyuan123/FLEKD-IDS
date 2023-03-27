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
        # out = torch.nn.functional.softmax(self.out(out), dim=1)
        out = self.out(out)
        return out

    def predict(self, X):
        _, predicted = torch.max(F.softmax(self.forward(X), dim=1), 1)
        return predicted