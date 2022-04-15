class TrajectoryStep:
    def __init__(self, column, logProbAction, reward, applyResult):
        self.column = column
        self.logProbAction = logProbAction
        self.reward = reward
        self.applyResult = applyResult
