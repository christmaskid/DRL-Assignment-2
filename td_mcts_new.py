# ==== VERSION: DEALING WITH STOCHASITY BY EXPANDING TWO LAYERS OF NODES. VER.2 ==== #

import copy
import random
import math
import numpy as np

MIN_RAND_NODES=4
Q_NORM_VALUE=1 #30 #1 #0.5

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, env, state, score, action=None, parent=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action

        self.visits = 0
        self.total_reward = 0.0

        self.children = {}
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0 and len(self.children)>0

class After_Node:
    def __init__(self, afterstate, score, action=None, parent=None):
        self.afterstate = afterstate
        self.score = score
        self.parent = parent
        self.action = action

        self.visits = 0
        self.total_reward = 0.0

        self.children = {}
        self.untried_actions_num = min(len(np.where(afterstate==0)[0]), MIN_RAND_NODES)

    def fully_expanded(self):
      return self.untried_actions_num==0 and len(self.children)>0



class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_random_tile(self, node):
        empty_cells = list(zip(*np.where(node.afterstate == 0)))
        if not empty_cells:
          return None
        x, y = random.choice(empty_cells)
        tile = 2 if random.random() < 0.9 else 4
        return (x,y,tile)

    def select_child(self, node):
        if isinstance(node, TD_MCTS_Node):
          best_score = -float('inf')
          selected_child = None

          q_values = dict()
          for child in node.children.values():
            if child.visits > 0:
              q_values[child] = child.total_reward
          q_values_avg = np.mean(list(q_values.values()))
          min_q, max_q = min(q_values.values()), max(q_values.values())

          for child in node.children.values():
            if child.visits == 0:
              return child
              # q = q_values_avg
            else:
              q = q_values[child]

            # Normalize
            if max_q==min_q:
              q = 0
            else:
              q = (q-min_q)/(max_q-min_q) * Q_NORM_VALUE

            uct_value = q + self.c * math.sqrt(math.log(node.visits) / child.visits)
            # print("q", q, "explore_term", uct_value-q, "uct", uct_value)

            print(child.action, child.visits, self.approximator.value(child.afterstate)+child.score, child.total_reward, uct_value, flush=True)
            
            if uct_value > best_score:
              best_score, selected_child = uct_value, child

          selected_action = selected_child.action
          print()

        else:
        #   best_score = -float('inf')
        #   selected_child = random.choice(list(node.children.values()))
        
          keys = node.children.keys()
          twos = [(i,j,tile) for (i,j,tile) in keys if tile==2]
          fours = [(i,j,tile) for (i,j,tile) in keys if tile==4]
          if len(twos) == 0:
            selected_action = random.choice(fours)
          elif len(fours) == 0:
            selected_action = random.choice(twos)
          elif random.random() < 0.9:
            selected_action = random.choice(twos)
          else:
            selected_action = random.choice(fours)
          selected_child = node.children[selected_action]

        # print(node, selected_child)#, selected_action)
        return selected_child

    # def rollout(self, sim_env, depth): # random rollout
    #     afterstate = sim_env.board
    #     # print("init", self.approximator.value(sim_env.board), sim_env.score, flush=True)
    #     while depth>0:
    #       legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
    #       if not legal_moves:
    #         break
    #       action = random.choice(legal_moves)
    #       _, afterstate, _, _, _ = sim_env.step(action)
    #       depth -= 1
    #     # print("rollout", self.approximator.value(sim_env.board), sim_env.score, flush=True)
    #     # print(self.approximator.value(sim_env.board), np.max(sim_env.board))
    #     return self.approximator.value(sim_env.board) + sim_env.score
    #     # return self.approximator.value(afterstate) + sim_env.score

    def rollout(self, sim_env, depth): # greedy rollout
      done = False
    #   print("init", self.approximator.value(sim_env.board), sim_env.score, flush=True)
      while depth > 0:
          legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
          if not legal_moves:
              break
          best_score = -float('inf')
          best_action = None
          for a in legal_moves:
              test_env = copy.deepcopy(sim_env)
              _, afterstate, next_score, _, _ = test_env.step(a)
              reward = next_score - sim_env.score
              value = reward + self.approximator.value(afterstate)
              if value > best_score:
                  best_score = value
                  best_action = a
          _, _, _, done, _ = sim_env.step(best_action)
          depth -= 1
        #   print("rollout", depth, self.approximator.value(sim_env.board), sim_env.score, flush=True)
    #   print("rollout", self.approximator.value(sim_env.board), sim_env.score, flush=True)
      value_est = self.approximator.value(afterstate) + sim_env.score
      return value_est


    def backpropagate(self, node, reward):
        # print("backprop", reward, end=" ")
        while node:
          # print(" {}({})->".format(node.total_reward, node.visits), end="")
          node.visits += 1
          node.total_reward += (reward - node.total_reward) / node.visits
          # print("{}({}), ".format(node.total_reward, node.visits), end="")
          # reward *= self.gamma # decreasing
          node = node.parent
        # print("\n")

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # Selection
        done = False
        while node.fully_expanded():
          node = self.select_child(node)
          # print("Select", node, node.action)
          if isinstance(node, After_Node):
            _, afterstate, next_score, done, _ = sim_env.step(node.action)
          else:
            i, j, tile = node.action
            afterstate = sim_env.board.copy()
            afterstate[i, j] = tile
            sim_env.board = afterstate.copy()
          if done:
            break

        # Expand
        if not node.fully_expanded() and not done:
          if isinstance(node, TD_MCTS_Node) and len(node.untried_actions)>0: # select an action
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            _, afterstate, next_score, _, _ = sim_env.step(action)
            node.children[action] = After_Node(afterstate=afterstate.copy(), score=next_score, parent=node, action=action)
            # print("Expand an action:", node, action, node.children[action])
            node = node.children[action]

          elif isinstance(node, After_Node) and node.untried_actions_num>0: # add an random tile
            action = self.select_random_tile(node)
            node.untried_actions_num -= 1
            i, j, tile = action
            sim_env.board[i, j] = tile
            node.children[action] = TD_MCTS_Node(self.env, state=sim_env.board.copy(), score=sim_env.score, parent=node, action=action)
            # print("Expand a random tile:", node, action, node.children[action])
            node = node.children[action]

        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # rollout_reward = self.approximator.value(sim_env.board) + sim_env.score
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_action = max(root.children, key=lambda a: root.children[a].visits, default=None)

        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0

        return best_action, distribution
