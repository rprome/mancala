import sys
import time
import copy

class MancalaGame:
    def __init__(self, state_str):
        """Initialize the game from the given state string."""
        parts = state_str.split()
        self.num_pits = int(parts[1])  # N = 6
        self.board = list(map(int, parts[2:2 + 2 * self.num_pits + 2]))  # Board state
        self.turn = int(parts[2 + 2 * self.num_pits + 2])  # Turn count
        self.current_player = int(parts[3 + 2 * self.num_pits + 2])  # Player (1 or 2)
        self.start_time = time.time()  # Track computation time

    def get_valid_moves(self):
        """Return a list of valid moves for the current player."""
        start = 0 if self.current_player == 1 else self.num_pits + 1
        return [i for i in range(start, start + self.num_pits) if self.board[i] > 0]

    def make_move(self, pit):
        """Execute a move and return whether the player gets an extra turn."""
        stones = self.board[pit]
        self.board[pit] = 0
        index = pit

        while stones > 0:
            index = (index + 1) % (2 * self.num_pits + 2)
            if (self.current_player == 1 and index == self.num_pits + 1) or (self.current_player == 2 and index == self.num_pits):
                continue  # Skip opponent's store
            self.board[index] += 1
            stones -= 1

        # Capture rule
        if (self.current_player == 1 and 0 <= index < self.num_pits) or (self.current_player == 2 and self.num_pits + 1 <= index < 2 * self.num_pits + 1):
            if self.board[index] == 1:
                opposite_index = (2 * self.num_pits) - index
                if self.board[opposite_index] > 0:
                    store_index = self.num_pits if self.current_player == 1 else (2 * self.num_pits + 1)
                    self.board[store_index] += self.board[opposite_index] + 1
                    self.board[index] = 0
                    self.board[opposite_index] = 0

        # Check for extra turn
        store_index = self.num_pits if self.current_player == 1 else (2 * self.num_pits + 1)
        return index == store_index

    def is_game_over(self):
        """Check if one side has no stones left."""
        player1_side = sum(self.board[:self.num_pits]) == 0
        player2_side = sum(self.board[self.num_pits + 1:-1]) == 0
        return player1_side or player2_side

    def apply_pie_rule(self):
        """Apply the PIE rule by swapping sides if Player 2 chooses."""
        self.board = self.board[self.num_pits + 1:] + self.board[:self.num_pits + 1]  # Swap
        self.current_player = 1  # Player 2 becomes Player 1

    def evaluate_board(self):
        """Simple heuristic: difference in store stones."""
        return self.board[self.num_pits] - self.board[-1]

    def minimax(self, depth, alpha, beta, maximizing):
        """Minimax with Alpha-Beta Pruning."""
        if depth == 0 or self.is_game_over() or (time.time() - self.start_time > 0.9):  # Ensure 1 sec limit
            return self.evaluate_board(), None

        valid_moves = self.get_valid_moves()
        best_move = None

        if maximizing:  # AI's turn (maximize score)
            max_eval = float('-inf')
            for move in valid_moves:
                new_game = copy.deepcopy(self)
                extra_turn = new_game.make_move(move)
                eval_score, _ = new_game.minimax(depth - 1, alpha, beta, extra_turn or not maximizing)
                if eval_score > max_eval:
                    max_eval, best_move = eval_score, move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Pruning
            return max_eval, best_move

        else:  # Opponent's turn (minimize AI's advantage)
            min_eval = float('inf')
            for move in valid_moves:
                new_game = copy.deepcopy(self)
                extra_turn = new_game.make_move(move)
                eval_score, _ = new_game.minimax(depth - 1, alpha, beta, extra_turn or maximizing)
                if eval_score < min_eval:
                    min_eval, best_move = eval_score, move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Pruning
            return min_eval, best_move

    def play_turn(self):
        """Determines and prints the best move for the current player."""
        if self.turn == 2 and self.current_player == 2:
            print("PIE")  # If Player 2, can choose PIE rule
            return

        _, best_move = self.minimax(6, float('-inf'), float('inf'), True)
        if best_move is None:
            best_move = self.get_valid_moves()[0]  # Fallback to a valid move
        print(best_move + 1)  # Convert to 1-based indexing


if __name__ == "__main__":
    input_state = sys.stdin.read().strip()
    game = MancalaGame(input_state)
    game.play_turn()
