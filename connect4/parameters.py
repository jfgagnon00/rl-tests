class Parameters:
    # Board and rules parameters
    WinningStreak = 4
    BoardWidth = 7
    BoardHeight = 6

    # Model parameters
    LearningRate = 1e-4
    Gamma = 0.9
    OpenAIState = True

    # Training parameters
    Episodes = 20000
    LogEpisodeEveryN = 1000
    SaveEveryN = 2000
    EpsilonThreshold = 0.8
    EpsilonDecay = 0.99
