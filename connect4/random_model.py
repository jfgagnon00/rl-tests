import torch

class RandomModel(torch.nn.Module):
    def __init__(self, device, numOutputs):
        super().__init__()
        self._probabilities = torch.full((1, numOutputs), 1.0 / numOutputs, dtype=torch.float32)
        self._probabilities.to(device)
        self.numOutputs = numOutputs

    def forward(self, x):
        return self._probabilities
