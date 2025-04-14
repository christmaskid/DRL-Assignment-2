# Remember to adjust your student ID in meta.xml
import pickle

from approximator import NTupleApproximator
from libenv2048.env2048 import Game2048Env
from td_mcts_new import *

import sys
sys.modules['__main__'].NTupleApproximator = NTupleApproximator
patterns = [
        # https://ko19951231.github.io/2021/01/01/2048/
        ((1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)),
        ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)),
        ((0, 1), (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)),
    ]
approximator = NTupleApproximator(board_size=4, patterns=patterns)
approximator = pickle.load(open("approximator_4_6_my_new.pkl", "rb"))

def get_action(state, score):
    env = Game2048Env()
    env.board = state
    env.score = score
        
    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=0, gamma=0.99)
    root = TD_MCTS_Node(env, state, score)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)
    print(env.board, "\nScore", env.score, flush=True)
    print(best_act, _, flush=True)
    return best_act