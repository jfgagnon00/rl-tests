import numpy as np
import torch

from model_base import *


class SimpleModel(ModelBase):
    def __init__(self, device, numInputs, numOutputs, hiddenLayersNumFeatures, numHiddenLayers):
        super().__init__(device, numInputs, numOutputs)

        layers = []
        for i in range(numHiddenLayers):
            layers.append( torch.nn.Linear(numInputs, hiddenLayersNumFeatures) )
            layers.append( torch.nn.ReLU() )
            # layers.append( torch.nn.BatchNorm1d(hiddenLayersNumFeatures) )
            numInputs = hiddenLayersNumFeatures
        layers.append( torch.nn.Linear(numInputs, numOutputs) )
        layers.append( torch.nn.Softmax(-1) )

        self._model = torch.nn.Sequential(*layers)
        self._model.to(device)
        self._model.share_memory()

    def forward(self, cells):
        return self._model(cells)
