import torch

from model_base import *


class RandomModel(ModelBase):
    def __init__(self, device, numInputs, numOutputs):
        super().__init__(device, numInputs, numOutputs)

        self._probabilities = torch.full((1, numOutputs), 1.0 / numOutputs, dtype=torch.float32)
        self._probabilities.to(device)
        self._probabilities.share_memory_()

    def forward(self, x):
        return self._probabilities
