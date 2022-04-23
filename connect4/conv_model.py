import torch.nn

from model_base import *


class ConvModel(ModelBase):
    def __init__(self, device, numInputs, numOutputs):
        super().__init__(device, numOutputs)

        layers = []
        layers.append( torch.nn.Conv2d(1, 24, 5, padding=2) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Conv2d(24, 1, 5, padding=2) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Flatten(0) )
        layers.append( torch.nn.Linear(numInputs, numOutputs) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Softmax(-1) )

        self._model = torch.nn.Sequential(*layers)
        self._model.to(device)
        self._model.share_memory()

    def forward(self, cells):
        # cells is assumed to be a 2D tensor having numInputs elements
        shape = cells.shape
        x = cells.reshape(1, 1, shape[0], shape[1])
        return self._model(x)
