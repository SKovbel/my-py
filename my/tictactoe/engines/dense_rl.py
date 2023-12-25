import numpy as np  # linear algebra
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


class EngineDenseRL:
    unique = False
    unique_ids = {}

    EPOCHS = 30

    REWARD_K = 0.9
    REWARD_WIN = 0.90
    REWARD_DRW = 0.10
    REWARD_LST = 0.00
    REWARD_ILG = -1.00

    def __init__(self, code='EngineDenseRL', unique=False):
        self.code = code
        self.unique = unique
        self.model = Sequential([
            layers.ReLU(32, input_shape=(9,)),
            #layers.Normalization(),
            layers.ReLU(32),
            #layers.Dropout(0.2),
            layers.Normalization(),
            layers.Dense(9, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='mse')   # binary_crossentropy
        self.model.summary()

        checkpoint_filepath = 'model_checkpoint.h5'
        self.checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,  # Set to True if you only want to save the model weights
            save_freq='epoch',  # Set to 'epoch' to save at the end of every epoch
            period=100  # Save every 100 epochs
        )

    def init(self, game):
        pass

    def move(self, game):
        x = game.board()
        y = self.model.predict([x], verbose=0)
        y = game.remove_illegal_moves(y)
        best_move = np.argmax(y)
        return best_move

    def train(self, game):
        # this engine trains only on the end of game
        if game.is_tie():
            return

        runs = self.check_uniq(game)
        #if self.check_uniq(game):
        #    return

        turn = game.player
        reward_a = self.REWARD_WIN if game.status() == 1 else (self.REWARD_DRW if game.status() == 0 else self.REWARD_LST)
        reward_b = self.REWARD_LST if game.status() == 1 else (self.REWARD_DRW if game.status() == 0 else self.REWARD_WIN)

        timeseries = game.timeseries()[:-1]     # skip last series
        targets = self.model.predict(timeseries, verbose=0)
        for t in range(len(timeseries) - 1, 0, -1):
            move = game.moves[t]
            if turn == 1:
                reward_a = targets[t][move] + self.REWARD_K * (reward_a - targets[t][move])
                targets[t][move] = reward_a
            elif turn == -1:
                reward_b = targets[t][move] + self.REWARD_K * (reward_b - targets[t][move])
                targets[t][move] = reward_b
            turn = -turn

        #early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        self.model.fit(np.array(timeseries), np.array(targets), epochs=self.EPOCHS - runs, verbose=0)
                       #callbacks=[self.checkpoint_callback]
                       #, callbacks=[early_stopping])

    def check_uniq(self, game):
        if self.unique:
            game_id = ''.join(str(item) for item in game.moves)
            if game_id not in self.unique_ids:
                self.unique_ids[game_id] = 0
            self.unique_ids[game_id] += 1
            return self.unique_ids[game_id]
        return 0
