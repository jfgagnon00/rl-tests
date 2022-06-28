import torch
import numpy as np

from algorithm import *
from rules import *

class DQN(Algorithm):

    def __init__(self, parameters):
        self._parameters = parameters

    def eval(self, cells, model, color):
        pass

    def backward(self, simulationTrajectory):
        pass

    def update(self):
        pass

    def _sample(self, cells, model, color):
        pass