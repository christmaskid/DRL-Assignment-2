import copy
import random
import math
import numpy as np

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, env, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
    
    def fully_expanded(self):
        return len(self.untried_actions) == 0 and len(self.children) > 0

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

    def select_child(self, node):
        best_child = None
        best_score = -float('inf')
        
        for child in node.children.values():
            if child.visits > 0:
                uct_value = (child.total_reward / child.visits) + self.c * math.sqrt(math.log(max(1, node.visits)) / (child.visits + 1))
            else:
                uct_value = float('inf')

            if uct_value > best_score:
                best_score = uct_value
                best_child = child
        
        return best_child

    def rollout(self, sim_env, depth):
        afterstate = sim_env.board.copy()
        while depth > 0:
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            _, afterstate, _, _, _ = sim_env.step(action)
            depth -= 1
        return self.approximator.value(afterstate)

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += self.gamma * reward
            reward = self.gamma * reward
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        while node.fully_expanded():
            node = self.select_child(node)
            _, next_state, next_score, done, _ = sim_env.step(node.action)

            if done:
                break
            
            empty_cells = [(i, j) for i in range(4) for j in range(4) if next_state[i, j] == 0]
            if empty_cells:
                i, j = random.choice(empty_cells)
                tile_value = 2 if random.random() < 0.9 else 4
                next_state[i, j] = tile_value
                node.children[(i, j, tile_value)] = TD_MCTS_Node(self.env, state=next_state.copy(), score=next_score, parent=node)
            
            node = self.select_child(node)

        if node.untried_actions:
            action = random.choice(node.untried_actions)
            node.untried_actions.remove(action)
            next_state, _, next_score, _, _ = sim_env.step(action)
            node.children[action] = TD_MCTS_Node(self.env, state=next_state, score=next_score, parent=node, action=action)
            node = node.children[action]

        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_action = max(root.children, key=lambda a: root.children[a].visits, default=None)
        
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
        
        return best_action, distribution
