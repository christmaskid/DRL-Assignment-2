import os, sys
import numpy as np
import math, random
import copy

class Connect6PUCTNode:
    def __init__(self, untried_actions, state, score, prior, parent=None, action=None):
        self.state = state # board
        self.score = score # score of the board
        self.prior = prior
        self.parent = parent # None for root
        self.action = action # parent -> action (by player) -> me

        self.untried_actions = untried_actions
        self.children = {}
        
        # Rollout/backprop related
        self.visits = 0
        self.total_reward = 0.0

    def fully_expanded(self):
        return len(self.untried_actions) == 0
    
class Connect6PUCTMCTS:
    def __init__(self, env, iterations, exploration_constant, rollout_depth):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth

    def create_env_from_state(self, state, score, depth):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        new_env.depth = depth # to determine who the current player 
        return new_env

    def select_child(self, node):
        selected_child = None
        best_score = -float('inf')

        for child in node.children.values():
            puct_value = child.total_reward + self.c * child.prior \
                        * math.sqrt(math.log(node.visits)) / (child.visits+1)
            if selected_child is None or puct_value > best_score:
                selected_child = child
                best_score = puct_value

        return selected_child
    
    def rollout(self, sim_env, rollout_depth):
        
        while rollout_depth>0:
            legal_actions = sim_env.get_legal_actions()
            if not legal_actions: break
            # action = random.choice(legal_actions) # ?
            best_score = -float('inf')
            best_action = None
            for action, prior in legal_actions:
                test_env = copy.deepcopy(sim_env)
                _, score, _, _ = test_env.step(action)
                if score + prior > best_score:
                    best_action, best_score = action, score + prior
            sim_env.step(best_action)
            rollout_depth -= 1
        
        return sim_env.score
    
    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.total_reward += (reward - node.total_reward) / node.visits
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score, depth=0)

        # 1. Selection
        done = False
        while node.fully_expanded() and not done:
            node = self.select_child(node)
            _, _, done, _ = sim_env.step(node.action)

        # 2. Expansion
        if len(node.untried_actions)>0:
            action, prior = random.choice(node.untried_actions)
            node.untried_actions.remove((action, prior))
            legal_actions = sim_env.get_legal_actions()
            node.children[action] = Connect6PUCTNode(
                                    untried_actions=legal_actions,
                                    state = sim_env.board.copy(), score = sim_env.score,
                                    prior = prior,
                                    parent = node, action = action)
            node = node.children[action]

        # 3. Rollout
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # 4. Backprop
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        best_visits = -1
        best_action = None
    
        distribution = []
        for action, child in root.children.items():
            visit_prob = child.visits / total_visits if total_visits > 0 else 0
            distribution.append(visit_prob)
            
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        
        return best_action, distribution