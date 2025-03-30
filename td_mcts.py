import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, env, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        # List of untried actions based on the current state's legal moves
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        selected_child = None
        best_score = -float('inf')
        uct_values = dict()

        for child in node.children.values():
          if child.visits > 0:
            uct_values[child] = child.total_reward + self.c * math.sqrt(math.log(node.visits) / child.visits)
        average = sum(uct_values.values()) / len(uct_values)
        # print(uct_values.values(), average)

        for child in node.children.values():
          if child.visits == 0:
            uct_value = average
          else:
            uct_value = uct_values[child]
          
          if selected_child is None or uct_value > best_score:
            best_score = uct_value
            selected_child = child
        return selected_child

    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        value_est = 0
        while depth > 0:
          legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
          if len(legal_moves) == 0:
            break # leaf; game over
          action = random.choice(legal_moves)
          _, reward,_, _ = sim_env.step(action)
          depth -= 1
          value_est += self.gamma * reward

        # TODO: Use the approximator to evaluate the final state.
        value_est += self.approximator.value(sim_env.board) # instead of sim_env.score
        return value_est


    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node:
          node.visits += 1
          node.total_reward += (reward - node.total_reward) / node.visits
          node = node.parent


    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded():
          node = self.select_child(node)
          _, _, done, _ = sim_env.step(node.action)

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if len(node.untried_actions) > 0:
          action = random.choice(node.untried_actions)
          node.untried_actions.remove(action)
          next_state, next_score, done, _ = sim_env.step(action)
          node.children[action] = TD_MCTS_Node(self.env, state=next_state, score=next_score, parent=node, action=action)
          node = node.children[action]


        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
