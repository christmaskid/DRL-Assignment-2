import sys
import copy
from tqdm import tqdm
import numpy as np
# from lib_connect6_env.mcts_connect6_compiled import Connect6UCTNode, Connect6UCTMCTS
# from lib_connect6_env.connect6_env_compiled import Connect6GameEnv
from lib_connect6_env.mcts_connect6_new import Connect6UCTNode, Connect6UCTMCTS
# from lib_connect6_env.connect6_two_stage_env import Connect6GameEnv
# from lib_connect6_env.connect6_env import Connect6GameEnv
from lib_connect6_env.connect6_env_new_new import Connect6GameEnv

class Connect6Game:
    def __init__(self, size=19, turn=1):
        self.env = Connect6GameEnv(size=size, opp_env=None)
        self.turn = turn

    def reset_board(self):
        self.env.reset()
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.env.set_board_size(size)
        print("= ", flush=True)

    def check_win(self):
        return self.env.check_win()
    
    def play_move(self, color, move):
        """Places stones and checks the game status."""
        self.env.play_move(color, move)
        print('= ', end='', flush=True)

    def generate_move(self, color):
        """Generates a move using UCT-MCTS."""
        if self.env.game_over:
            print("? Game over")
            return
        
        sim_env = copy.deepcopy(self.env)
        self.turn = 1 if color.upper() == 'B' else 2
        print("I am player {}, color {}".format(self.turn, color), flush=True, file=sys.stderr)
    
        # Initialize MCTS with current game state
        uct_mcts = Connect6UCTMCTS(
            env=sim_env,
            exploration_constant=1.414,  # UCT exploration parameter
            iterations=10,  # Can increase due to smaller action space
            rollout_depth=4,  # Can increase due to faster iterations
        )
        
        # Create root node with current state
        time = -1 if np.sum(self.env.board)==0 else 0
        legal_moves = self.env.get_legal_actions(self.turn)
        root = Connect6UCTNode(
            untried_actions=legal_moves, 
            state=self.env.board.copy(), 
            score={1:0, 2:0},
            player=(1,time) if color.upper() == 'B' else (2,time)
        )
        self.env.show_board_err(legal_moves)
        
        # Run MCTS simulations
        for _ in tqdm(range(uct_mcts.iterations)):
            # print("Iteration", _, flush=True, file=sys.stderr)
            uct_mcts.run_simulation(root)
        
        # Select best move based on visit counts
        print(root.children.keys(), flush=True, file=sys.stderr)
        selected, distribution = uct_mcts.best_action_distribution(root)
        print(distribution, flush=True, file=sys.stderr)
        print("selected", selected, flush=True, file=sys.stderr)
        r, c = selected
        move_str = f"{self.env.index_to_label(c)}{r+1}"
        
        self.env.play_move(color, move_str)
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr, flush=True)
        return

    def show_board(self):
        self.env.show_board()
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print(flush=True)
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            # except Exception as e:
            #     print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
