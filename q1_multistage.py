import copy
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import sys, os
import random

# from env2048 import Game2048Env
from libenv2048.env2048compiled import Game2048Env

# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------

def rot90(coords):
  return tuple((3-y, x) for (x, y) in coords)

def rot180(coords):
  return tuple((3-x, 3-y) for (x, y) in coords)

def rot270(coords):
  return tuple((y, 3-x) for (x, y) in coords)

def flip(coords):
  return tuple((x, 3-y) for (x, y) in coords)

def rot90_flip(coords):
  return tuple((3-y, 3-x) for (x, y) in coords)

def rot180_flip(coords):
  return tuple((3-x, y) for (x, y) in coords)

def rot270_flip(coords):
  return tuple((y, x) for (x, y) in coords)


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        syms = [pattern]
        for func in [
            rot90, rot180, rot270, flip,
            rot90_flip, rot180_flip, rot270_flip
        ]:
            syms.append(func(pattern))
        return syms


    def tile_to_index(self, tile):
        """
        Converts tile values to xan index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)


    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        value_sum = 0
        for i in range(len(self.patterns)):
          for j in range(8):
            pattern = self.symmetry_patterns[i*8+j]
            feature = self.get_feature(board, pattern)
            value_sum += self.weights[i][feature]
        return value_sum

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i in range(len(self.patterns)):
          for j in range(8):
            pattern= self.symmetry_patterns[i*8+j]
            feature = self.get_feature(board, pattern)

            # Optimistic Initialization
            if feature not in self.weights[i]:
              if random.random() < 0.1:
                 self.weights[i][feature] = 1e4
              else:
                 self.weights[i][feature] = 0
            self.weights[i][feature] += alpha * delta / len(self.symmetry_patterns)


def get_stage_num(max_tile):
    if max_tile >= 2**16:
        return 3
    elif max_tile >= 2**13:
        return 2
    return 1


def td_learning(env, approximator, approximators_dict, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1, start_episode=0):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    all_stage_num = max(approximators_dict.keys())
    all_max_tile = 1
    final_scores = []
    success_flags = []
    
    mode = 0 # 0: training, 1: collection
    samples = []

    for episode in range(start_episode, num_episodes):
        state = env.reset()
        if len(samples) > 0:
           sample = random.choice(samples)
           state.board, state.score = sample # initialize the next stage map with previous stage sample

        trajectory = []  # Store trajectory data if needed
        previous_score = 0
        done = False
        max_tile = np.max(state)
        stage_num = all_stage_num # start from current training state
        approximator = approximators_dict[stage_num]

        while not done:
            approximator = approximators_dict[stage_num] #pickle.load(open(f"approximator_stage{stage_num}.pkl", "rb"))
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break
            # TODO: action selection
            # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
            action_values = []
            for action in legal_moves:
                sim_env = copy.deepcopy(env)
                next_state, reward, _, _ = sim_env.step(action)
                action_values.append(approximator.value(next_state))
            action = legal_moves[np.argmax(action_values)]

            cur_state = np.copy(state)
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            if mode == 0:
              stage_num = get_stage_num(max_tile)
              if stage_num not in approximators_dict:
                approximators_dict[stage_num] = NTupleApproximator(board_size=4, patterns=patterns)
              approximator = approximators_dict[stage_num]

              # Freeze all the other weights, just update the max one
              if stage_num >= all_stage_num:
                trajectory.append((cur_state, action, np.copy(next_state), incremental_reward, done))
              
            elif mode == 1:
               samples.append((next_state.copy(), new_score))

            state = next_state

        # Freeze all the other weights, just update the max one
        if mode == 0:
          approximator = approximators_dict[all_stage_num]
          for t in reversed(range(len(trajectory))):
            state, action, next_state, incremental_reward, done = trajectory[t]
            # print(next_state, "\n", state)
            if done:
              delta = incremental_reward - approximator.value(state)
            else:
              delta = incremental_reward + gamma * approximator.value(next_state) - approximator.value(state)
            approximator.update(state, delta, alpha)


        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        all_max_tile = max(all_max_tile, max_tile)
        all_stage_num = get_stage_num(all_max_tile)

        sep = 100 #0
        if (episode + 1) % sep == 0:
          avg_score = np.mean(final_scores[-sep:])
          success_rate = np.sum(success_flags[-sep:]) / sep
          print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Max tile: {all_max_tile} | Mode: {mode}", flush=True)
          print(approximators_dict, flush=True)
          
          pickle.dump(approximators_dict[all_stage_num], open(f"approximator_stage{all_stage_num}.pkl", "wb"))
          # pickle.dump(approximators_dict, open("approximator_final.pkl", "wb"))

          if len(final_scores)>1000:
            moving_avg = np.mean(final_scores[:1000])
            if abs(moving_avg-avg_score)<200:
              mode = 1 # collecting samples
          


    return final_scores


if __name__=="__main__":
    # TODO: Define your own n-tuple patterns
    patterns = [
        # https://ko19951231.github.io/2021/01/01/2048/
        ((1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)),
        ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)),
        ((1, 2), (1, 3), (2, 2), (2, 3), (3, 2), (3, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1)),
        ((0, 1), (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)),
    ]

    
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    approximators_dict = {1: approximator}
    if len(sys.argv)>1 and os.path.exists(sys.argv[1]):
        try:
           for key in [1,]:
              approximators_dict[key] = pickle.load(open("approximator_stage{}.pkl".format(key), "rb"))
        except:
          approximators_dict = pickle.load(open(sys.argv[1], "rb"))
    start_episode = eval(sys.argv[2]) if len(sys.argv)>2 else 0

    env = Game2048Env()

    # Run TD-Learning training
    # Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
    # However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
    final_scores = td_learning(env, approximator, approximators_dict, num_episodes=100000, alpha=0.1, gamma=0.99, epsilon=0.1, start_episode=start_episode)
    pickle.dump(approximators_dict, open("approximator_final.pkl", "wb"))

    plt.plot(final_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.savefig("n-tuple_td-learning_stages.png")
