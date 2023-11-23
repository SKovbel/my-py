import os
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model


# 0 1 2
# 3 4 5
# 6 7 8
class TicTacToe:
    moves = []
    fields = []

    def __init__(self):
        pass

    def auto_play(self, playerX, player0):
        self.new_game(playerX, player0)

        for i in range(0, 9):
            if self.auto_move() == None:
                continue
            print("Won ", self.status)
            break

    def new_game(self, engineX, engine0):
        self.status = None
        self.moves = []
        self.fields = [0, 0, 0,  0, 0, 0,  0, 0, 0]
        self.engineX = engineX
        self.engine0 = engine0

    def who_next(self):
        return 1 if len(self.moves) % 2 == 0 else -1

    def move(self, position):
        pass

    def auto_move(self):
        if self.status == None: # game is not finished
            who = self.who_next()
            best_move = self.engineX.move(self) if who == 1 else self.engine0.move(self)
            self.moves.append(best_move)
            self.fields[best_move] = who

        self.status = self.check_status()
        if self.status != None:
            self.engineX.end_game(self)
            self.engine0.end_game(self)

        return self.status

    # <1> if won X, <-1> if won 0, <0> if draw <None> game is not finished
    def check_status(self, fields = None):
        rules = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # horz
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # vert
            [0, 4, 8], [2, 4, 6] # diag
        ]

        isDraw = True
        fields = fields if fields else self.fields
        for rule in rules:
            absSum = abs(fields[rule[0]] + fields[rule[1]] + fields[rule[2]])
            if (absSum == 3):
                return fields[rule[0]] # return who won (1, -1)
            if (isDraw):
                sumAbs = abs(fields[rule[0]]) + abs(fields[rule[1]]) + abs(fields[rule[2]])
                isDraw = False if absSum == sumAbs else isDraw
        return 0 if isDraw else None # return draw or game is not finished
    

class EngineMinMiax:
    inf = 1000

    def __init__(self, code = 'MinMax', max_depth = 5):
        self.code = code
        self.max_depth = max_depth

    def move(self, game):
        best_fields = self.minimax(game, game.fields)
        best_move = random.randint(0, len(best_fields))
        return best_fields[best_move]

    def end_game(self, game):
        pass

    def minimax(self, game, fields, turn = 1, depth = 0):
        if depth:
            status = game.check_status(fields);
            if status != None:
                return status / depth # win

            if depth >= self.max_depth:
                return 0

        best_score = -self.inf if turn > 0 else self.inf
        best_moves = []

        for i in range(len(fields)):
            if fields[i] != 0: # skip zero squares
                continue
            # only 1 and -1
            fields[i] = turn
            score = self.minimax(game, fields, -turn, depth + 1)
            if turn > 0 and score >= best_score:
                if best_score != score:
                    best_moves = []
                best_score = score
                if depth == 0:
                    best_moves.append(i)
            elif turn < 0 and score <= best_score:
                if (best_score != score):
                    best_moves = []
                best_score = score
                if depth == 0:
                    best_moves.append(i)
            fields[i] = 0

        return best_score if depth else best_moves


class TicTacToeLayer(layers.Layer):
    def __init__(self, multiplier = 0, **kwargs):
        super(TicTacToeLayer, self).__init__(**kwargs)
        self.multiplier = multiplier
        print('TicTacToeLayer init')

    def call(self, inputs):
        moves = 9 * [0]
        X = inputs[1:]
        time_series = []
        print('TicTacToeLayer a1')
        turn = 1
        print(inputs, X.shape)
        for m in X:
            print(m)
            moves[m] = turn
            exit(0)
            time_series.append(moves.copy())
            turn = -turn
        print('TicTacToeLayer a2')
        time_series = time_series.reshape(time_series.shape[0], time_series.shape[1], 1)
        print('TicTacToeLayer a3')
        return time_series


class TicTacToeModel(Model):
    def __init__(self):
        print('TicTacToeModel init')
        super(TicTacToeModel, self).__init__()
        self.__layers = [
            #TicTacToeLayer(),
            layers.LSTM(9, activation='relu', input_shape=(9, 9)),
            layers.Dense(9)
        ]

    def call(self, inputs):
        X = inputs
        for i in range(len(self.__layers)):
            X = self.__layers[i](X)
        return X

    def moves_to_time(self, X):
        turn = 1
        moves = 9 * [0]
        time_series = [moves]
        for m in range(len(X)):
            moves[m] = turn
            time_series.append(moves.copy())
            turn = -turn
        time_series = np.array(time_series)
        print(time_series)
        time_series = time_series.reshape(time_series.shape[0], time_series.shape[1], 1)
        return time_series

# standard approach
def createModel():
    model = Sequential([
        TicTacToeLayer(),
        layers.LSTM(9, activation='relu', input_shape=(9, 9)),
        layers.Dense(9)
    ])
    model.compile(optimizer='adam', loss='mse')  # Use mean squared error for regression
    return model

# todo try id of engine as feature
class EngineAiMl:
    REWARD_K = 0.9
    REWARD_WIN = 0.90
    REWARD_DRW = 0.40
    REWARD_LST = 0.10
    REWARD_ILG = 0.01

    def __init__(self, code = 'AiMl'):
        self.code = code
        self.model = TicTacToeModel()
        #self.model = createModel()

    def move(self, game):
        X = self.model.moves_to_time(game.moves)
        Y = self.model.predict(X)
        best_move = np.where(arr == 3)[0]Y.index(max(Y))
        return best_move

    def end_game(self, game):
        X = game.moves
        Y = self.model.predict(X)
        Y = self.rewards(X, Y)
        self.model.fit(X, Y, epochs=100, verbose=1)

    def rewards(self, X, Y):
        rewardA = REWARD_WIN if abs(game.status) != 0 else REWARD_DRW     # last move win if (game status = -1 or 1) or draw (if 0).
        rewardB = REWARD_LST if abs(game.status) != 0 else REWARD_DRW     # prev move lost if (game status = -1 or 1) or draw

        last_player = true
        for h in reversed(X):
            last_player = not last_player

            moveIdx = X[h]
            X[moveIdx] = 0

            if playerLast: # calculating X and 0 separatly
                rewardA = Y[moveIdx] + REWARD_K * (rewardA - Y[moveIdx])
                Y[moveIdx] = rewardA
            elif not playerLast:
                rewardB = Y[moveIdx] + REWARD_K * (rewardB - Y[moveIdx])
                Y[moveIdx] = rewardB

            # illegal moves has own rewards
            for i in range(len(Y)):
                Y[i] = Y[i] if Y[i] == 0 else REWARD_ILG

        return X, Y


game = TicTacToe()
engines = [EngineMinMiax('minmax'), EngineAiMl('aiml')]

for i in range(10):
    engineX, engineY = random.sample(engines, k=2)
    game.auto_play(engineX, engineY)
