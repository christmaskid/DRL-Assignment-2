# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

from q1 import NTupleApproximator
from libenv2048.env2048compiled import Game2048Env

import sys
sys.modules['__main__'].NTupleApproximator = NTupleApproximator
patterns = [
        # https://ko19951231.github.io/2021/01/01/2048/
        ((1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)),
        ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)),
        ((1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)),
        ((0, 1), (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)),
    ]
approximator = NTupleApproximator(board_size=4, patterns=patterns)
approximator = pickle.load(open("approximator_new.pkl", "rb"))

def get_action(state, score):
    env = Game2048Env()
    env.board = state
    env.score = score
    
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    # shall not be empty

    # TODO: Use your N-Tuple approximator to play 2048
    action_values = []
    for action in legal_moves:
        sim_env = copy.deepcopy(env)
        _, afterstate, _, _, _ = sim_env.step(action)
        action_values.append(approximator.value(afterstate))
    best_action = legal_moves[np.argmax(action_values)]


    return best_action
