class Rules:
    ColorNone = 0
    ColorBlack = -1
    ColorRed = 1

    def colorName(color):
        names = ["-", "Red", "Black"]
        return names[int(color)]

    ApplyInvalid = 0
    ApplyInconclusive = 1
    ApplyTie = 2
    ApplyWonVertical = 3
    ApplyWonHorizontal = 4
    ApplyWonDiag1 = 5
    ApplyWonDiag2 = 6

    def applyName(apply):
        names = ["Invalid", "Inconclusive", "Tie", "Won Vert", "Won Horiz", "Won Diag 1", "Won Diag 2"]
        return names[apply]

    def __init__(self, winningStreakLength):
        if winningStreakLength < 2:
            raise "Invalid streak length"
        self._winningStreakLength = winningStreakLength

    def apply(self, board, column, color):
        applyResult = Rules.ApplyInvalid

        if (0 <= column and column < board.width):
            row = board.heights[column]
            if row < board.height:
                board.cells[row, column] = color
                board.heights[column] = row + 1
                applyResult = self._hasWon(board, column)

                board.occupiedCells += 1
                if applyResult == Rules.ApplyInconclusive and board.isFull():
                    applyResult = Rules.ApplyTie

        return applyResult

    def _hasWon(self, board, column):
        row = board.heights[column] - 1

        # vertical
        if (self._isWinningStreak(board, row, column, 1, 0)):
            return Rules.ApplyWonVertical

        # horizontal
        if (self._isWinningStreak(board, row, column, 0, 1)):
            return Rules.ApplyWonHorizontal

        # diagonal 1
        if (self._isWinningStreak(board, row, column, 1, 1)):
            return Rules.ApplyWonDiag1

        # diagonal 2
        if (self._isWinningStreak(board, row, column, 1, -1)):
            return Rules.ApplyWonDiag2

        return Rules.ApplyInconclusive

    def _isWinningStreak(self, board, row, col, dr, dc):
        streakColor = board.cells[row, col]

        while True:
            r = row - dr
            c = col - dc
            if r < 0 or r >= board.height or c < 0  or c >= board.width:
                break

            color = board.cells[r, c]
            if color != streakColor:
                break

            row = r
            col = c

        streakLength = 1
        while True:
            r = row + dr
            c = col + dc
            if r < 0 or r >= board.height or c < 0  or c >= board.width:
                break

            color = board.cells[r, c]
            if color != streakColor:
                break

            row = r
            col = c
            streakLength += 1

        return streakLength >= self._winningStreakLength
