from cv2 import Algorithm

from connect4.reinforce import Reinforce


class Parameters:
    # Board and rules parameters
    WinningStreak = 4
    BoardWidth = 7
    BoardHeight = 6

    # Model parameters
    ModelClass = SimpleModel
    ModelParams = {
        "numInputs": Parameters.BoardWidth * Parameters.BoardWidth * 3,
        "numOutputs": Parameters.BoardWidth,
        "hiddenLayersNumFeatures": 30,
        "numHiddenLayers": 3,
    }

    # Algorithm parameters
    AlgorithmClass = Reinforce
    AlgorithmParams = {
        "Gamma": 0.9
    }

    # Training parameters
    LearningRate = 1e-4
    Episodes = 20000
    LogEpisodeEveryN = 1000
    SaveEveryN = 2000
    EpsilonThreshold = 0.8
    EpsilonDecay = 0.99
