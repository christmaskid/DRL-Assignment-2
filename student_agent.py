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
approximator = pickle.load(open("submission/Q3/approximator_4_6_my_new.pkl", "rb"))

def get_action(state, score):
    print("Approx", approximator.value(state), score, flush=True)
    env = Game2048Env()
    env.board = state
    env.score = score
        
    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=1.41 * 0.1, rollout_depth=0, gamma=0.99)
    root = TD_MCTS_Node(env, state, score)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    for child in root.children.values():
        q = child.total_reward
        if child.visits == 0:
            print(child.action,"q", q, 0)
            afterstate = child.state
        else:
            uct_value = q + td_mcts.c * math.sqrt(math.log(root.visits) / child.visits)
            print(child.action, child.score, "total_reward (q)", q, "explore_term", uct_value-q, "uct", uct_value)
            test_env = td_mcts.create_env_from_state(child.afterstate, child.score)
            _, afterstate, _, done, _ = test_env.step(child.action)
            print("afterstate", child.action, "\n", afterstate)
            print("approx", approximator.value(afterstate), done)
            
        print("td choice value", child.action, approximator.value(afterstate) + child.score)

        for grandchild in child.children.values():
            print(grandchild.action, end=" ")
        print()


    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)
    print(env.board, "\nScore", env.score, flush=True)
    print(best_act, _, flush=True)
    return best_act



# Approx 13819.517009951702 5844
# 0 q 21746.367384998353 -> 1 explore_term 0.4111884402736905 uct 21746.778573438627 -> 1.41
# 1 q 13953.972208943396 -> 0 explore_term 1.3944078435924894 uct 13955.366616786989 -> 1.39
# Approx 29860.85746170198 908
# [[ 2  2 16 32]
#  [ 8 16 32 64]
#  [ 4  8 16 32]
#  [ 2  4  8 16]]
# Score 908
# 3 [0.   0.   0.08 0.92]
"""
Approx 13251.624305283207 2616
(0, 3, 2)
[[  2   4  64   2]
 [256  16  32   8]
 [ 64   4  16   2]
 [  4   2   4   2]]
Score 2616
0 [0.92 0.08 0.   0.  ]
Agent finished in 222 steps, Score: 2620

[TD-MCTS]
0 total_reward (q) 17249.587902423576 explore_term 0.4111884402736905 uct 17249.99909086385
1 total_reward (q) 13854.122155882706 explore_term 1.3944078435924894 uct 13855.516563726298

[PURE TD]
0 td choice value 0 12274.252376059136
1 td choice value 1 13920.384764671197
"""