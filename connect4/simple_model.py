import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, device, numInputs, numOutputs, hiddenLayersNumFeatures, numHiddenLayers):
        super().__init__()

        layers = []
        layers.append( torch.nn.Flatten(0) )
        layers.append( torch.nn.Linear(numInputs, hiddenLayersNumFeatures) )
        layers.append( torch.nn.ReLU() )
        for i in range(numHiddenLayers):
            layers.append( torch.nn.Linear(hiddenLayersNumFeatures, hiddenLayersNumFeatures) )
            layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Linear(hiddenLayersNumFeatures, numOutputs) )
        layers.append( torch.nn.Softmax(-1) )

        self._device = device
        self._model = torch.nn.Sequential(*layers)
        self._model.to(device)

        self._probabilities = torch.full((1, numOutputs), 1.0 / numOutputs, dtype=torch.float32)

        self.numOutputs = numOutputs

    def forward(self, cells):
        if cells.device != self._device:
            cells = cells.to(self._device)
        return self._model(cells)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        data = torch.load(filename)
        self.load_state_dict(data)
        self.eval()
