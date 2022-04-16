import torch.nn

class ConvModel(torch.nn.Module):
    def __init__(self, numInputs, numOutputs):
        super().__init__()

        self._bla = torch.nn.Conv2d(1, 8, 5, padding=2)

        layers = []
        layers.append( torch.nn.Conv2d(1, 8, 5, padding=2) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Conv2d(8, 1, 5, padding=2) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Flatten(0) )
        layers.append( torch.nn.Linear(numInputs, numOutputs) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Softmax(-1) )
        self._model = torch.nn.Sequential(*layers)
        self.numOutputs = numOutputs

    def forward(self, cells):
        # cells is assumed to be a 2D tensor having numInputs elements
        shape = cells.shape
        x = cells.float().reshape(1, 1, shape[0], shape[1])
        return self._model(x)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        data = torch.load(filename)
        self.load_state_dict(data)
        self.eval()
