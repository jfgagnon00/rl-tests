import torch.nn

class ConvModel(torch.nn.Module):
    def __init__(self, device, numInputs, numOutputs):
        super().__init__()

        layers = []
        layers.append( torch.nn.Conv2d(1, 24, 5, padding=2) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Conv2d(24, 1, 5, padding=2) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Flatten(0) )
        layers.append( torch.nn.Linear(numInputs, numOutputs) )
        layers.append( torch.nn.ReLU() )
        layers.append( torch.nn.Softmax(-1) )

        self._device = device
        self._model = torch.nn.Sequential(*layers)
        self._model.to(device)

        self.numOutputs = numOutputs

    def forward(self, cells):
        # cells is assumed to be a 2D tensor having numInputs elements
        shape = cells.shape
        x = cells.reshape(1, 1, shape[0], shape[1])

        if x.device != self._device:
            x = x.to(self._device)

        return self._model(x)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        data = torch.load(filename)
        self.load_state_dict(data)
        self.eval()
