class TicTacToe2x(TicTacToe):
    IN_FEATURES = 2*9

    def board(self, pos=None):
        board = self.IN_FEATURES * [0, 0]
        pos = len(self.moves) if pos is None else pos
        last = pos if pos >= 0 else len(self.moves) + pos
        turn = 1
        for i in self.moves[0:last]:
            idx = 2 * i
            board[idx] = 1 if turn == 1 else 0
            board[idx + 1] = 1 if turn == -1 else 0
            turn = -turn
        return board

    def timeseries(self):
        moves = self.IN_FEATURES * [0., 0.]
        time_series = [moves.copy()]
        turn = 1
        for i in self.moves:
            move_idx = 2 * i
            moves[move_idx] = 1. if turn == 1 else 0.
            moves[move_idx + 1] = 1. if turn == -1 else 0.
            time_series.append(moves.copy())
            turn = -turn
        return time_series
x