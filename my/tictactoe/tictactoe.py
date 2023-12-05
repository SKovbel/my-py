import random


class TicTacToe:
    IN_FEATURES = 9
    RULES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # vertical
        [0, 4, 8], [2, 4, 6]  # diagonal
    ]

    def __init__(self):
        self.player = 1
        self.moves = []
        self.game_status = None
        self.engine_X = None
        self.engine_0 = None

    def start(self, engine_X=None, engine_0=None):
        self.player = 1
        self.moves = []
        self.game_status = None
        if engine_X and engine_0:
            self.engine_X = engine_X
            self.engine_0 = engine_0
        else:
            self.engine_X, self.engine_0 = random.sample([self.engine_X, self.engine_0], k=2)

        self.engine_X.init(self)
        self.engine_0.init(self)

    # <1> if won X, <-1> if won 0, <0> if draw
    def status(self, board=None):
        if board is None:
            return self.game_status

        is_draw = True
        for rule in self.RULES:
            abs_sum = abs(board[rule[0]] + board[rule[1]] + board[rule[2]])
            if abs_sum == 3:
                return board[rule[0]]  # return player won (1, -1)
            if is_draw:
                sum_abs = abs(board[rule[0]]) + abs(board[rule[1]]) + abs(board[rule[2]])
                is_draw = False if abs_sum == sum_abs else is_draw
        return 0 if is_draw else None  # return draw or game is not finished

    def board(self, pos=None):
        board = self.IN_FEATURES * [0]
        pos = len(self.moves) if pos is None else pos
        last = pos if pos >= 0 else len(self.moves) + pos
        turn = 1
        for idx in self.moves[0:last]:
            board[idx] = turn
            turn = -turn
        return board

    def timeseries(self):
        moves = self.IN_FEATURES * [0]
        time_series = [moves.copy()]
        turn = 1
        for move_idx in self.moves:
            moves[move_idx] = 1 if turn == 1 else -1
            time_series.append(moves.copy())
            turn = -turn
        return time_series

    def is_tie(self, status=None):
        status = status if status else self.game_status
        return status is None

    def is_game_over(self, status=None):
        status = status if status else self.game_status
        return status is not None

    def is_draw(self, status=None):
        status = status if status else self.game_status
        return status == 0

    def is_win(self, status=None):
        status = status if status else self.game_status
        return status == 1 or status == -1

    def is_legal_move(self, pos):
        return pos not in self.moves and 0 <= pos < 9

    def remove_illegal_moves(self, Y):
        for y in Y:
            for pos in self.moves:
                y[pos] = 0
            return Y

    def move(self, move_idx):
        if not self.is_legal_move(move_idx):
            print('Illegal move', move_idx, self.moves)
            return False
        self.moves.append(move_idx)
        # update game status
        self.game_status = self.status(self.board())
        # update player if game is not finished
        if self.is_tie():  # tie
            self.player = -self.player
        return True

    def auto_move(self):
        if self.is_tie():  # game is not finished
            best_move = self.engine_X.move(self) if self.player == 1 else self.engine_0.move(self)
            self.move(best_move)

    def auto_play(self):
        for i in range(0, 9):
            self.auto_move()
            if self.is_game_over():
                self.engine_X.train(self)
                self.engine_0.train(self)
                break

    def print(self):
        board = self.board()
        print(board)
        print("\n", self.engine_X.code, self.engine_0.code)
        for r in range(0, 3):
            for c in range(0, 3):
                mark = ' '
                mark = 'X' if board[3 * r + c] == 1 else mark
                mark = '0' if board[3 * r + c] == -1 else mark
                print(mark, end='')
            print()
