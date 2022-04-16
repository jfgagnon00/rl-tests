import torch

class SimpleModel(torch.nn.Module):
    def __init__(self, numInputs, numOutputs, hiddenLayersNumFeatures, numHiddenLayers):
        super().__init__()

        layers = []
        layers.append( torch.nn.Linear(numInputs, hiddenLayersNumFeatures) )
        layers.append( torch.nn.ReLU() )
        for i in range(numHiddenLayers):
            layers.append( torch.nn.Linear(hiddenLayersNumFeatures, hiddenLayersNumFeatures) )
            layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Linear(hiddenLayersNumFeatures, numOutputs) )
        layers.append( torch.nn.Softmax(-1) )

        self._layers = torch.nn.ModuleList(layers)
        self.numOutputs = numOutputs

    def forward(self, cells):
        x = cells.flatten(0).float()
        for l in self._layers:
            x = l(x)
        return x

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        data = torch.load(filename)
        self.load_state_dict(data)
        self.eval()
