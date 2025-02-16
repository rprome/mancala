"""
CSC 242 Project 1: Source Code
Sanchit Somani, Rizouana Prome, Shafayet Fahim, Emmanuel Ahishakiye
February 16th, 2025 at 11:59 PM

This .py file serves as the source code for our Mancala player. It works in tandem
with random_player.py and controller.py (provided by Adam Purtee).
The Mancala player uses a heuristic minimaxing algorithm to determine the best pit
to choose out of a selection of 6 pits. In order to work strictly within a provided
time constraint, the algorithm goes down the game tree using Iterative Deepening
Search, allowing it to leave with a save state immediately. In order to reach as
deep as possible, the algorithm uses alpha-beta pruning, dramatically increasing the
efficiency of this algorithm and avoiding exploration of unnecessary nodes.
"""
# Necessary libraries
import sys
import math
import time

# We're using this global variable to determine maximum depth for optimization.
global_depth = 0

'''
Game Logic
'''

# This function is the basis for the game and will run after a pit is chosen.
def apply_move(p1_pits, p2_pits, p1_store, p2_store, current_player, chosen_pit):
    # Taking parameters and assigning them to symbolic parts of the Mancala board.
    N = len(p1_pits)
    p1_pits = p1_pits[:]
    p2_pits = p2_pits[:]

    # This block of code runs if the current player is P1.
    if current_player == '1':
        seeds = p1_pits[chosen_pit - 1]
        p1_pits[chosen_pit - 1] = 0
    else:
        seeds = p2_pits[chosen_pit - 1]
        p2_pits[chosen_pit - 1] = 0
    current_pit = chosen_pit
    side = current_player

    # This while loop will run while there are seeds in-hand.
    while seeds > 0:
        # We set side to match what our current_player is.
        if side == '1':
            if current_pit < N: current_pit += 1
            elif current_pit > N:
                if current_player == '1':
                    p1_store += 1
                    seeds -= 1
                    if seeds == 0: return p1_pits, p2_pits, p1_store, p2_store, '1' # Extra Move
                side = '2'
                current_pit = 0
                continue
            if current_pit <= N:
                # Continue emptying hand.
                p1_pits[current_pit - 1] += 1
                seeds -= 1
                # check capture if last stone lands in an empty pit on your side
                if seeds == 0 and current_player == '1' and p1_pits[current_pit - 1] == 1:
                    opposite_pit = N - current_pit + 1
                    captured = p2_pits[opposite_pit - 1]
                    if captured > 0:
                        p2_pits[opposite_pit - 1] = 0
                        p1_store += (captured + 1)
                        p1_pits[current_pit - 1] = 0
        elif side == '2':
            if current_pit < N: current_pit += 1
            else:
                if current_player == '2':
                    p2_store += 1
                    seeds -= 1
                    if seeds == 0: return p1_pits, p2_pits, p1_store, p2_store, '2' # Extra move
                side = '1'
                current_pit = 0
                continue
            if current_pit <= N:
                p2_pits[current_pit - 1] += 1
                seeds -= 1
                # We're checking here to see if the opposite side has >1 when we have 0 at index(current_pit).
                if seeds == 0 and current_player == '2' and p2_pits[current_pit - 1] == 1:
                    opposite_pit = N - current_pit + 1
                    # Correction back to zero-based index from our one-based index.
                    captured = p1_pits[opposite_pit - 1]
                    # Assuming we capture something...
                    if captured > 0:
                        p1_pits[opposite_pit - 1] = 0
                        p2_store += (captured + 1)
                        p2_pits[current_pit - 1] = 0

    # In the case that we are not getting an extra turn. It is now the opponent's turn.
    next_player = '2' if (current_player == '1') else '1'
    return p1_pits, p2_pits, p1_store, p2_store, next_player

# On player 2's first turn, they can invoke PIE. In this case, the pits and stores will flip.
def apply_PIE(p1_pits, p2_pits, p1_store, p2_store):
    return p2_pits[:], p1_pits[:], p2_store, p1_store

# We should be checking to see if the game has met termination conditions after every turn.
def game_is_over(p1_pits, p2_pits): return sum(p1_pits) == 0 or sum(p2_pits) == 0

# When the game ends, all seeds still in the pits should be distributed to their respective stores.
def finalize_game(p1_pits, p2_pits, p1_store, p2_store):
    if sum(p1_pits) == 0:
        p2_store += sum(p2_pits)
        p2_pits = [0] * len(p2_pits)
    elif sum(p2_pits) == 0:
        p1_store += sum(p1_pits)
        p1_pits = [0] * len(p1_pits)
    return p1_pits, p2_pits, p1_store, p2_store

'''
Helper Functions
'''

# Determines if a pit can be chosen. It cannot be chosen when it is empty.
def get_valid_moves(p1_pits, p2_pits, turn, current_player):
    N = len(p1_pits)
    valid = []
    if turn == 2 and current_player == '2': valid.append("PIE")
    if current_player == '1':
        for i in range(N):
            if p1_pits[i] > 0: valid.append(i + 1)
    else:
        for i in range(N):
            if p2_pits[i] > 0: valid.append(i + 1)
    return valid

# Determines how many seeds are gained by a player when they capture one of the opponent's pit.
def one_move_capture_size(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice):
    new_p1_pits, new_p2_pits, new_p1_store, new_p2_store, next_player = apply_move(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice)
    if player == '1':
        captured_p2 = sum(p2_pits) - sum(new_p2_pits)
        return max(0, captured_p2)
    else:
        captured_p1 = sum(p1_pits) - sum(new_p1_pits)
        return max(0, captured_p1)

# Determines if the next player will be the same player, which would mean the move gained us an extra turn.
def one_move_earns_extra_turn(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice):
    new_p1_pits, new_p2_pits, new_p1_store, new_p2_store, next_player = apply_move(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice)
    return next_player == player

'''
Heuristic
'''

# Evaluates score without using PIE. If PIE is possible, evaluates score with PIE and takes the better route.
def evaluate_PIE(p1_pits, p2_pits, p1_store, p2_store, current_player, turn):
    score_without_PIE = evaluate_position(p1_pits, p2_pits, p1_store, p2_store, current_player, turn)
    if current_player == '2' and turn == 2:
        swapped_p1, swapped_p2, swapped_s1, swapped_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
        score_with_PIE = evaluate_position(swapped_p1, swapped_p2, swapped_s1, swapped_s2, current_player, turn)
        return max(score_without_PIE, score_with_PIE)
    else:
        return score_without_PIE

def evaluate_position(p1_pits, p2_pits, p1_store, p2_store, perspective, turn):
    """
    Your original scoring logic that factors in store difference, side difference,
    potential captures, etc. (without worrying about PIE).
    """
    # Example: your current logic
    W_STORE_DIFF = 5.0
    W_EXTRA_MOVE = 1
    W_BIG_STEAL  = 3
    W_SIDE_DIFF  = 0.5

    if perspective == '1':
        store_diff = p1_store - p2_store
        side_diff  = sum(p1_pits) - sum(p2_pits)
    else:
        store_diff = p2_store - p1_store
        side_diff  = sum(p2_pits) - sum(p1_pits)

    # Potential capture or extra move analysis, etc.
    # ...
    # Suppose you computed these:
    extra_move_bonus = 0
    steal_bonus = 0.0
    # (or your existing logic)

    score = (W_STORE_DIFF * store_diff) + (W_SIDE_DIFF * side_diff)
    score += extra_move_bonus + steal_bonus
    return score

################################################################################
# Alpha-Beta with Transposition Table
################################################################################

# Global transposition cache: key -> (value, best_move, depth)
transposition_table = {}

def alpha_beta(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
               alpha, beta, depth, start_time, time_limit, root_player):
    """
    Alpha-beta search (single perspective approach, always "max" for cur_player).
    Returns (best_value, best_move).
    - root_player is the original perspective (for the heuristic).
    """
    # Time check
    if (time.time() - start_time) >= time_limit:
        # Evaluate with the heuristic from root_player's perspective
        val = evaluate_PIE(p1_pits, p2_pits, p1_store, p2_store, root_player, turn)
        return val, None

    # If game over or depth=0, finalize & evaluate
    if game_is_over(p1_pits, p2_pits) or depth == 0:
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = evaluate_PIE(tmp1, tmp2, s1, s2, root_player, turn)
        return val, None

    moves = get_valid_moves(p1_pits, p2_pits, turn, cur_player)
    if not moves:
        # no moves => finalize & evaluate
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = evaluate_PIE(tmp1, tmp2, s1, s2, root_player, turn)
        return val, None

    # Transposition table lookup
    board_key = (tuple(p1_pits), tuple(p2_pits), p1_store, p2_store, turn, cur_player, depth)
    if board_key in transposition_table:
        cached_val, cached_move, cached_depth = transposition_table[board_key]
        if cached_depth >= depth:
            return cached_val, cached_move

    # Basic move ordering: sort by potential captures (descending)
    def move_priority(m):
        if m == "PIE":
            return -1  # put PIE near the front or back arbitrarily; you can tweak
        else:
            return one_move_capture_size(p1_pits, p2_pits, p1_store, p2_store, cur_player, m)
    moves.sort(key=move_priority, reverse=True)

    best_val = -math.inf
    best_move = None

    for move in moves:
        if move == "PIE":
            # apply PIE => sides swap
            new_p1, new_p2, new_s1, new_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
            # Turn used, next player is '1', so turn += 1
            nxt_player = '1'
            nxt_turn = turn + 1

            val, _ = alpha_beta(new_p1, new_p2, new_s1, new_s2,
                                nxt_turn, nxt_player,
                                alpha, beta, depth - 1,
                                start_time, time_limit,
                                root_player)

        else:
            # normal pit move
            pit = move
            new_p1, new_p2, new_s1, new_s2, nxt_player = apply_move(
                p1_pits, p2_pits, p1_store, p2_store, cur_player, pit
            )
            nxt_turn = turn if nxt_player == cur_player else (turn + 1)

            val, _ = alpha_beta(new_p1, new_p2, new_s1, new_s2,
                                nxt_turn, nxt_player,
                                alpha, beta, depth - 1,
                                start_time, time_limit,
                                root_player)

        if val > best_val:
            best_val = val
            best_move = move

        alpha = max(alpha, best_val)
        if alpha >= beta:
            break  # alpha-beta cutoff

    # Store in transposition table
    transposition_table[board_key] = (best_val, best_move, depth)

    return best_val, best_move

def alpha_beta_search(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player, time_limit=0.5):
    """
    Iterative deepening alpha-beta to find the best move for `cur_player`.
    We define root_player = cur_player so we evaluate from that perspective.
    """
    start_time = time.time()
    best_move = None
    max_depth = 25  # Increase as desired

    # We'll do iterative deepening from depth=1.max_depth
    global global_depth
    for depth in range(1, max_depth + 1):
        if (time.time() - start_time) >= time_limit:
            break
        global_depth+=1
        val, move = alpha_beta(
            p1_pits, p2_pits, p1_store, p2_store,
            turn, cur_player,
            -math.inf, math.inf,
            depth,
            start_time, time_limit,
            root_player=cur_player
        )
        # If still have time, update best_move
        if (time.time() - start_time) < time_limit and move is not None:
            best_move = move
        else:
            break
    print(f"Max depth reached: {global_depth}", file=sys.stderr)  # Print to stderr to avoid interfering with the game
    return best_move

'''
main
'''
def main():
    line = sys.stdin.readline().strip()
    # Example:
    # STATE 6 4 4 4 4 4 4 4 4 4 4 4 4 0 0 1 1
    parts = line.split()
    N = int(parts[1])
    p1_pits = list(map(int, parts[2: 2 + N]))
    p2_pits = list(map(int, parts[2 + N: 2 + 2*N]))
    p1_store = int(parts[2 + 2*N])
    p2_store = int(parts[3 + 2*N])
    turn = int(parts[-2])
    cur_player = parts[-1]  # '1' or '2'

    # Clear the transposition table each run
    transposition_table.clear()

    # Find best move
    best_move = alpha_beta_search(
        p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
        time_limit=0.5
    )

    # If for some reason no best move was found, pick a fallback
    if not best_move:
        valid_moves = get_valid_moves(p1_pits, p2_pits, turn, cur_player)
        if valid_moves:
            best_move = valid_moves[0]
        else:
            best_move = 1

    print(best_move)

if __name__ == "__main__":
    main()