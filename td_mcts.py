import copy
import random
import math
import numpy as np
MAX_RAND_NODES=4

class TD_MCTS_Node:
    def __init__(self, env, state, score, action=None, parent=None):
        self.state = state.copy()
        self.score = score
        self.parent = parent
        self.action = action

        self.visits = 0
        self.total_reward = 0.0

        self.children = {}
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]

    def fully_expanded(self):
        return len(self.untried_actions) == 0
    
class After_Node:
    def __init__(self, afterstate, score, action=None, parent=None):
        self.afterstate = afterstate
        self.score = score
        self.parent = parent
        self.action = action

        self.visits = 0
        self.total_reward = 0.0

        self.children = {}
        self.untried_action_num = min(len(np.where(afterstate==0)[0])*2, MAX_RAND_NODES)

    def fully_expanded(self):
        return self.untried_action_num==0
    
class TD_MCTS:
    def __init__(self, env, approximator, iterations, exploration_constant,
                 rollout_depth, gamma=0.99):
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
        empty_cells = list(zip(*np.where(node.afterstate==0)))
        if len(empty_cells)==0: # should not happen
            return None
        x, y = random.choice(empty_cells)
        tile = 2 if random.random() < 0.9 else 4
        return (x, y, tile)
    
    def select_child(self, node):
        if isinstance(node, TD_MCTS_Node):
            best_score = -float('inf')
            selected_child = None

            q_values = []
            for child in node.children.values():
                if child.visits == 0:
                    return child
                q_values.append(child.total_reward)
            min_q, max_q = min(q_values), max(q_values)

            for child in node.children.values():
                q = child.total_reward
                # here child.visits > 0 must held
                uct_value = q + self.c * math.sqrt(math.log(node.visits) / child.visits)
                if uct_value > best_score:
                    best_score, selected_child = uct_value, child

            return selected_child

        elif isinstance(node, After_Node):
            keys = node.children.keys()
            twos = [item for item in keys if item[2] == 2]
            fours = [item for item in keys if item[2] == 4]
            if len(twos) == 0:
                action = random.choice(fours)
            elif len(fours) == 0:
                action = random.choice(twos)
            elif random.random() < 0.9:
                action = random.choice(twos)
            else:
                action = random.choice(fours)
            return node.children[action]

        else:
            print("Wrong node type: {}".format(type(node)))
            raise Exception


    def rollout(self, env, depth, afterstate):
        # # here the state of env must be an afterstate
        # afterstate = env.board.copy()
        done = env.is_game_over()

        while depth>0 and not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if len(legal_moves)==0:
                done = True
                break
            action = random.choice(legal_moves)
            _, afterstate, _, done, _ = env.step(action, True)
            depth -= 1

        if done:
            return env.score
        else:
            return self.approximator.value(afterstate) + env.score
        
    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent

    def run_simulation(self, root):
        # assert isinstacne(root, TD_MCTS_Node)
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)
        # 1. Selection
        done = False
        while node.fully_expanded() and not done:
            node = self.select_child(node)
            if isinstance(node, TD_MCTS_Node):
                # Now place THE random tile on the board
                r, c, tile = node.action
                sim_env.board[r, c] = tile
                done = sim_env.is_game_over()
            elif isinstance(node, After_Node):
                # Now move the board according to the action
                _, afterstate, _, done, _ = sim_env.step(node.action, False)
            else:
                print("Wrong node type: {}".format(node), flush=True)
                raise Exception
        
        # 2. Expansion
         # TD_MCTS_Node -> action -> After_Node -> add random tile -> TD_MCTS_Node -> ...
        if not done and not node.fully_expanded():

            if isinstance(node, TD_MCTS_Node):
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                _, afterstate, next_score, done, _ = sim_env.step(action, False)
                node.children[action] = After_Node(afterstate, next_score, action=action, parent=node)
                node = node.children[action]
            
            elif isinstance(node, After_Node):
                action = self.select_random_tile(node)
                node.untried_action_num -= 1
                r, c, tile = action
                afterstate = sim_env.board.copy()
                sim_env.board[r, c] = tile
                done = sim_env.is_game_over()
                node.children[action] = TD_MCTS_Node(sim_env, sim_env.board.copy(), 
                                                     sim_env.score, action=action, parent=node)
                node = node.children[action]

        # 3. Rollout (or not) and 4. Backpropagation
        # if isinstance(node, After_Node):
        rollout_reward = self.rollout(sim_env, self.rollout_depth, afterstate)
        self.backpropagate(node, rollout_reward)
                
    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_action = max(root.children, key=lambda a: root.children[a].visits, default=None)

        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0

        return best_action, distribution
                
                