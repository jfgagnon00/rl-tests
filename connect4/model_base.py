import torch


class ModelBase(torch.nn.Module):
    def __init__(self, device, numOutputs):
        super().__init__()

        self.device = device
        self.numOutputs = numOutputs

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        data = torch.load(filename)
        self.load_state_dict(data)
        self.eval()
