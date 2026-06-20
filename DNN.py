import torch
import torch.nn as nn
class DNN(nn.Module):
    model = None

    def __init__(self, model):
        self.model = model
    
    def forward(self, input):
        self.model.eval()
        with torch.no_grad():
            return self.model(input)