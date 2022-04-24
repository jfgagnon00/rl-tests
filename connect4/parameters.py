class Parameters:
    # Board and rules parameters
    WinningStreak = 4
    BoardWidth = 6
    BoardHeight = 7

    # Model parameters
    LearningRate = 1e-3
    Gamma = 0.8

    # Training parameters
    Episodes = 100000
    LogEpisodeEveryN = 500
    SaveEveryN = 1000
    UseMultiprocessing = True
    NormalizeGradients = False
