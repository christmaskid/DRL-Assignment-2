import sys
import numpy as np
import random

class Connect6GameEnv:
    def __init__(self, first_player, size=19, board=None,  depth=0):
        self.size = size
        self.depth = depth
        if board is not None:
            self.board = board.copy()
        else:
            self.board = np.zeros((size, size), dtype=int)
        self.last_move = None
        self.game_over = self.is_game_over()
        self.score = self.evaluate_board(1)
        self.first_player = first_player


    def reset(self):
        self.board.fill(0)
        self.game_over = False
        self.score = 0
        self.last_move = None
        self.depth = 0
        return self.board
    
    def check_win(self):
        if not self.last_move: return 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
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
                # Check backward
                for i in range(1, 6):
                    rr, cc = r - dr * i, c - dc * i
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
            return 3 - self.first_player
        return self.first_player

    def get_legal_actions(self):
        player = self.convert_depth_to_player(self.depth)
        # empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r,c]==0]
        # return empty_positions
    
        # TODO: compress action space
        if np.sum(self.board) == 0:
            return [(self.size//2+1, self.size//2+1)] # center
        # Check for direct threats first (positions that could lead to >=4 connections)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1),
                      (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        threat_positions = []
        for r in range(self.size):
            for c in range(self.size):
                color = self.board[r, c]
                if color == 3-player:
                    for dr, dc in directions:
                        def legal_pos(nr,nc):
                            return (0<=nr and nr<=self.size-1 and 0<=nc and nc<=self.size-1)
                        # Look for sequences of same color that could be extended
                        count = 1
                        empty_spots = []
                        # Check forward
                        for i in range(1, 6):
                            rr, cc = r + dr * i, c + dc * i
                            if not (0 <= rr < self.size and 0 <= cc < self.size):
                                break
                            if self.board[rr, cc] == color:
                                count += 1
                            elif self.board[rr, cc] == 0 and \
                                (legal_pos(rr-dr, cc-dc) and self.board[rr-dr, cc-dc] != 0 or \
                                 legal_pos(rr+dr, cc+dc) and self.board[rr+dr, cc+dc] != 0):
                                empty_spots.append((rr, cc))
                            else:
                                break
                        if count >= 4 and empty_spots:
                            threat_positions.extend(empty_spots)

        if threat_positions:
            return list(set(threat_positions))
        
        positions = []
        radius = 1 
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
        for pos in positions:
            self.board[pos] = player
            score = self.evaluate_position(*pos, player)
            self.board[pos] = 0
            positions_scores_pairs.append((pos, score))
        positions_scores_pairs = sorted(positions_scores_pairs, key=lambda x: -x[1])
        positions = [item[0] for item in positions_scores_pairs]

        return positions[:min(10, len(positions))] if positions \
            else [(r, c) for r in range(self.size) 
                  for c in range(self.size) if self.board[r,c] == 0]


    def evaluate_position(self, r, c, color):
        """Evaluates the strength of a position based on alignment potential."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = 0

        for dr, dc in directions:
            count = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == color:
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == color:
                count += 1
                rr -= dr
                cc -= dc

            if count >= 5:
                score += 10000
            elif count == 4:
                score += 5000
            #elif count == 3:
            #    score += 1000
            #elif count == 2:
            #    score += 100
        
        return score
    
    def evaluate_board(self, turn):
        def count_patterns(player):
            score = 0
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            
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
                        
                        # Score different patterns
                        scoring_patterns = {
                            10000: ["xxxxxx"],
                            1000: ["0xxxxx", "xxxxx0"], 
                            100: ["0xxxx0", "00xxxx", "xxxx00", "x00xxx", "xx00xx", "xxx00x",
                                  "0x0xxx", "x0xxx0", "xx0xx0", "0xx0xx", "0xxx0x", "xxx0x0"], 
                            5: ["0xxx00", "00xxx0"], 
                        }
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