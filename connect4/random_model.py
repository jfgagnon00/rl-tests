import torch

class RandomModel(torch.nn.Module):
    def __init__(self, numOutputs):
        super().__init__()
        self._probabilities = torch.full((1, numOutputs), 1.0 / numOutputs, dtype=torch.float32)

    def forward(self, x):
        return self._probabilities
