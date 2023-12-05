import random


class EngineMinMiax:
    inf = 1000

    def __init__(self, code='MinMax', max_depth=5):
        self.code = f"{code}-{max_depth}"
        self.max_depth = max_depth

    def move(self, game):
        best_moves = self.minimax(game, game.board())
        best_move = random.randint(0, len(best_moves) - 1)
        return best_moves[best_move]

    def init(self, game):
        pass

    def train(self, game):
        pass

    def minimax(self, game, board, turn=1, depth=0):
        if depth:
            status = game.status(board)
            if game.is_game_over(status):
                return status / depth   # win

            if depth >= self.max_depth:
                return 0

        best_score = -self.inf if turn > 0 else self.inf
        best_moves = []

        for i in range(len(board)):
            if board[i] != 0:  # skip zero squares
                continue
            # only 1 and -1
            board[i] = turn
            score = self.minimax(game, board, -turn, depth + 1)
            if turn > 0 and score >= best_score:
                if best_score != score:
                    best_moves = []
                best_score = score
                if depth == 0:
                    best_moves.append(i)
            elif turn < 0 and score <= best_score:
                if best_score != score:
                    best_moves = []
                best_score = score
                if depth == 0:
                    best_moves.append(i)
            board[i] = 0

        return best_score if depth else best_moves
