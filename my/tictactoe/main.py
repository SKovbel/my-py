import random
from stats import Stats
from tictactoe import TicTacToe
from engines.minmax import EngineMinMiax
from engines.dense_rl import EngineDenseRL
from engines.dense_dqn import EngineDenseDQN

game = TicTacToe()
stats = Stats()
engines = [
    EngineMinMiax('EngineMinMiax', max_depth=5),
    # EngineDenseRL('EngineDenseRL', unique=False),
    EngineDenseDQN('EngineDenseDQN', unique=True)
]

for game_num in range(1, 2000):
    engine_X, engine_0 = random.sample(engines, k=2)
    game.start(engine_X, engine_0)
    game.auto_play()
    stats.add(game)
