import copy
import math
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import sys, os
import random
from tqdm import tqdm

# from env2048 import Game2048Env
from libenv2048.env2048compiled import Game2048Env
from q1 import *

def get_stage_num(max_tile):
    if max_tile >= 2**16:
        return 3
    elif max_tile >= 2**13:
        return 2
    return 1


def td_learning(env, approximator, approximators_dict, patterns,
                num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1, 
                start_episode=0, stage_num=1):
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
    
    samples = []

    while all_stage_num < 4:
        new_samples = []

        # 1. Collection: initial states of all_stage_num
        if all_stage_num >= 1:
            print("Collect samples at stage {}".format(all_stage_num), flush=True)
            
            for episode in tqdm(range(1000)):
                state = env.reset()
                if len(samples)>0: # initialize the next stage map with previous stage sample
                    sample = random.choice(samples)
                    env.board, env.score = sample 
                    state = env.board
                
                done = False
                max_tile = np.max(state)
                stage_num = get_stage_num(max_tile)
                
                while not done:
                    approximator = approximators_dict[stage_num] #pickle.load(open(f"approximator_stage{stage_num}.pkl", "rb"))
                    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
                    if not legal_moves:
                        break
                    action_values = []
                    for action in legal_moves:
                        sim_env = copy.deepcopy(env)
                        next_state, afterstate, reward, _, _ = sim_env.step(action)
                        action_values.append(reward + gamma * approximator.value(afterstate))
                    action = legal_moves[np.argmax(action_values)]
                    
                    next_state, _, new_score, done, _ = env.step(action)
                    stage_num = get_stage_num(max_tile)
                    if stage_num in approximators_dict:
                        approximator = approximators_dict[stage_num]

                    new_samples.append((next_state.copy(), new_score))
        
        samples = new_samples
        all_stage_num += 1
        approximators_dict[all_stage_num] = NTupleApproximator(board_size=4, patterns=patterns)
        print("Collect {} samples for stage {}.".format(len(samples), all_stage_num), flush=True)

        # 2. Update: for all_stage_num+1
        for episode in range(start_episode, num_episodes):
            state = env.reset()

            if len(samples)>0: # initialize the next stage map with previous stage sample
                sample = random.choice(samples)
                env.board, env.score = sample 
                state = env.board

            trajectory = []  # Store trajectory data if needed
            previous_score = 0
            done = False
            max_tile = np.max(state)
            stage_num = all_stage_num # start from current training state
            prev_afterstate = state.copy()
            approximator = approximators_dict[all_stage_num]

            while not done:
                approximator = approximators_dict[stage_num] #pickle.load(open(f"approximator_stage{stage_num}.pkl", "rb"))
                legal_moves = [a for a in range(4) if env.is_move_legal(a)]
                if not legal_moves:
                    break
                
                action_values = []
                for action in legal_moves:
                    sim_env = copy.deepcopy(env)
                    next_state, afterstate, reward, _, _ = sim_env.step(action)
                    action_values.append(reward + gamma * approximator.value(afterstate))
                action = legal_moves[np.argmax(action_values)]

                next_state,  afterstate, new_score, done, _ = env.step(action)
                incremental_reward = new_score - previous_score
                previous_score = new_score
                max_tile = max(max_tile, np.max(next_state))

                stage_num = get_stage_num(max_tile)
                if stage_num not in approximators_dict:
                    approximators_dict[stage_num] = NTupleApproximator(board_size=4, patterns=patterns)
                approximator = approximators_dict[stage_num]

                # Freeze all the other weights, just update the max one
                if stage_num >= all_stage_num:
                    trajectory.append((prev_afterstate.copy(), action, afterstate.copy(), incremental_reward, done))
                
                state = next_state
                prev_afterstate = afterstate

            # Freeze all the other weights, just update the max one
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

            sep = 100 #0
            if (episode + 1) % sep == 0:
                avg_score = np.mean(final_scores[-sep:])
                success_rate = np.sum(success_flags[-sep:]) / sep
                print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | Max tile: {all_max_tile}", flush=True)
                print(approximators_dict, flush=True)
                
                pickle.dump(approximators_dict[all_stage_num], open(f"approximator_stage{all_stage_num}.pkl", "wb"))
                # pickle.dump(approximators_dict, open("approximator_final.pkl", "wb"))
            
            all_max_tile = max(all_max_tile, max_tile)
            if get_stage_num(all_max_tile) > all_stage_num:
                print("Max tile {} reached, stage {} ends.".format(all_max_tile, all_stage_num), flush=True)
            
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
    final_scores = td_learning(env, approximator, approximators_dict, patterns,
                               num_episodes=25000, alpha=0.1, gamma=0.99, epsilon=0.1, 
                               start_episode=start_episode, stage_num=1)
    pickle.dump(approximators_dict, open("approximator_final.pkl", "wb"))

    plt.plot(final_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.savefig("n-tuple_td-learning_stages-new.png")
