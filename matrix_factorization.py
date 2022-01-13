import torch
from torch import nn
import torch.optim as optim

class MF(torch.nn.Module):
    def __init__(self, w, h, d):
        super(MF, self).__init__()
        self.w = w
        self.h = h
        self.d = d
        self.layer1 = torch.nn.Linear(w, d, bias=False)
        self.layer2 = torch.nn.Linear(d, h, bias=False)
        self.indicator = torch.eye(w)

    def forward(self):
        return self.layer2(self.layer1(self.indicator))

model = MF(100, 100, 10)
prediction = model()

matrix = torch.ones(100, 100)
loss = (matrix - prediction).pow(2).sum()

print()