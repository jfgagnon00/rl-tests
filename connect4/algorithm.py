from abc import ABC, abstractmethod

class Algorithm(ABC):
    @abstractmethod
    def eval(self, board, model, color):
        """
        Returns tuple:
            action (cpu readable)
            algorithm state (whatever object needed for backward to operate)
        """
        pass

    @abstractmethod
    def backward(self, simulationTrajectory):
        """
        Returns list of discounted return
        """
        pass

    @abstractmethod
    def update(self):
        pass