import sys
import numpy as np
import random


# Score different patterns
scoring_patterns = {
    100000: ["xxxxxx"],
    10000: ["0xxxxx", "xxxxx0", "xxxx0x", "xx0xxx", "xxx0xx", "xxx00xx", "xx0xxx"], 
    1000: ["0xxxx0", "00xxxx", "xxxx00", "x00xxx", "xx00xx", "xxx00x",
            "0x0xxx", "x0xxx0", "xx0xx0", "0xx0xx", "0xxx0x", "xxx0x0"], 
    50: ["0xxx00", "00xxx0"], 
    3: ["xxx", "xx0x", "x0xx"],
    2: ["0xx", "xx0", "x0x"],
    1: ["0x", "x0"],
}
directions = [(0, 1), (1, 0), (1, 1), (1, -1),
            (0, -1), (-1, 0), (-1, -1), (-1, 1)]

class Connect6GameEnv:
    def __init__(self, size=19, board=None,  depth=0):
        self.size = size
        self.depth = depth
        if board is not None:
            self.board = board.copy()
        else:
            self.board = np.zeros((size, size), dtype=int)
        self.last_move = None
        self.game_over = self.is_game_over()
        self.score = self.evaluate_board(1)


    def reset(self):
        self.board.fill(0)
        self.game_over = False
        self.score = 0
        self.last_move = None
        self.depth = 0
        return self.board
    
    def check_win(self):
        if not self.last_move: return 0
        positions = [pos for pos in self.last_move if pos]
        
        for r, c in positions:
            color = self.board[r, c]
            if color == 0:
                continue
                
            for dr, dc in directions:
                # Check 4 positions in both directions from current position
                count = 1
                # Check forward
                for i in range(1, 6):
                    rr, cc = r + dr * i, c + dc * i
                    if not (0 <= rr < self.size and 0 <= cc < self.size) or self.board[rr, cc] != color:
                        break
                    count += 1
                if count >= 6:
                    return color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def is_game_over(self):
        if np.all(self.board != 0) or self.check_win() != 0:
            return True
        return False
    
    def convert_depth_to_player(self, depth):
        if (depth-1)%4 in [1,2]:
            return 2
        return 1

    def get_legal_actions(self):
        player = self.convert_depth_to_player(self.depth)
        # empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r,c]==0]
        # return empty_positions

        positions = []
        radius = 2
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:  # if position is occupied
                    # Check surrounding positions within 'radius' steps
                    for dr in range(-radius, radius+1):
                        for dc in range(-radius, radius+1):
                            nr, nc = r + dr, c + dc
                            # Check if position is valid and empty
                            if (0 <= nr < self.size and 
                                0 <= nc < self.size and 
                                self.board[nr, nc] == 0 and 
                                (nr, nc) not in positions):
                                positions.append((nr, nc))
        
        positions_scores_pairs = []
        for (r, c) in positions:
            if self.board[r, c] != 0: continue # occupied
            score = max(
                self.evaluate_position(r, c, player),
                self.evaluate_position(r, c, 3-player)
            )
            positions_scores_pairs.append(((r,c), score))
        random.shuffle(positions_scores_pairs)
        positions_scores_pairs = sorted(positions_scores_pairs, key=lambda x: -x[1])
        # print("positions_scores_pairs", positions_scores_pairs, file=sys.stderr, flush=True)
        positions = [item[0] for item in positions_scores_pairs]

        return positions_scores_pairs[:min(10, len(positions))] \
            if positions \
            else [((r, c), 0) for r in range(self.size) 
                  for c in range(self.size) if self.board[r,c] == 0]


    def evaluate_position(self, r, c, player):
        """Evaluates the strength of a position based on alignment potential."""
        score = 0
        for dr, dc in directions:
            # Look ahead 6 positions
            line = []
            for i in range(-5, 6):
                rr, cc = r + dr * i, c + dc * i
                if 0 <= rr < self.size and 0 <= cc < self.size:
                    line.append(self.board[rr, cc])
                else:
                    line.append(-1)  # Out of bounds
            
            # Convert line to string for pattern matching
            pattern = ''.join(str(x) for x in line)
            for key in scoring_patterns:
                for scoring_pattern in scoring_patterns[key]:
                    if scoring_pattern.replace('x', str(player)) in pattern:
                        score += key
        return score
    
    def evaluate_board(self, turn):
        def count_patterns(player):
            score = 0            
            for r in range(self.size):
                for c in range(self.size):
                    if self.board[r, c] != player:
                        continue
                        
                    for dr, dc in directions:
                        # Look ahead 6 positions
                        line = []
                        for i in range(-5, 6):
                            rr, cc = r + dr * i, c + dc * i
                            if 0 <= rr < self.size and 0 <= cc < self.size:
                                line.append(self.board[rr, cc])
                            else:
                                line.append(-1)  # Out of bounds
                        
                        # Convert line to string for pattern matching
                        pattern = ''.join(str(x) for x in line)
                        for key in scoring_patterns:
                            for scoring_pattern in scoring_patterns[key]:
                                if scoring_pattern.replace('x', str(player)) in pattern:
                                    score += key
            
            return score
        
        if self.depth>0:  # Not the initial empty board
            player_score = count_patterns(turn)
            opponent_score = count_patterns(3 - turn)
            return player_score - opponent_score
        return 0
    

    def step(self, action):
        player = self.convert_depth_to_player(self.depth)
        self.board[action[0], action[1]] = player
        self.depth += 1
        self.score = self.evaluate_board(player)

        done = self.is_game_over()
        return self.board, self.score, done, {}
    
    
    def show_board_err(self, candidates=[]):
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 \
                                            else "O" if self.board[row, col] == 2 \
                                            else "*" if (row, col) in candidates \
                                            else "." for col in range(self.size))
            print(line, file=sys.stderr)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels, file=sys.stderr)