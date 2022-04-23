class Parameters:
    # Board and rules parameters
    WinningStreak = 4
    BoardWidth = 6
    BoardHeight = 7

    # Model parameters
    LearningRate = 1e-3
    Gamma = 0.8

    # Training parameters
    Episodes = 10
    LogEpisodeEveryN = 100
    SaveEveryN = 1000
    UseMultiprocessing = False
    NormalizeGradients = False
