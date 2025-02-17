"""
CSC 242 Project 1: Source Code
Authors: Sanchit Somani, Rizouana Prome, Shafayet Fahim, Emmanuel Ahishakiye
Submission Date: February 16th, 2025
Description: This .py file serves as the source code for a Mancala player that competes
against random_player.py using controller.py, both of which were provided by Professor
Adam Purtee. The source code uses heuristic minimaxing and alpha-beta pruning to determine
the optimal path to win the game against test cases RandomOpponent, MinimaxOpponent, and
AlphaBetaOpponent.
"""

# Necessary imports.
import sys
import math
import time

# Necessary global variables.
transposition_table = {}

"""
Section: Game Logic Functions
The following functions allow the program to manipulate seed counts and check
termination conditions for the game.
"""

# After a pit is chosen by our heuristic, change seed counts across the board.
def apply_move(p1_pits, p2_pits, p1_store, p2_store, current_player, chosen_pit):
    # Taking our input and modifying it to our needs.
    N = len(p1_pits)
    p1_pits = p1_pits[:]
    p2_pits = p2_pits[:]

    # Gets seed count for a pit based on player.
    if current_player == '1':
        seeds = p1_pits[chosen_pit - 1]
        p1_pits[chosen_pit - 1] = 0
    else:
        seeds = p2_pits[chosen_pit - 1]
        p2_pits[chosen_pit - 1] = 0

    # Based on player, set side and what pit we get.
    current_pit = chosen_pit
    side = current_player

    # Distribute seeds until our hand is empty. Checks conditionally who's playing.
    while seeds > 0:
        if side == '1':
            if current_pit < N: current_pit += 1
            elif current_pit == N:
                if current_player == '1':
                    p1_store += 1
                    seeds -= 1
                    if seeds == 0: return p1_pits, p2_pits, p1_store, p2_store, '1'
                side = '2'
                current_pit = 0
                continue
            if current_pit <= N:
                p1_pits[current_pit - 1] += 1
                seeds -= 1
                if seeds == 0 and current_player == '1' and p1_pits[current_pit - 1] == 1:
                    opposite_pit = N - current_pit + 1
                    captured = p2_pits[opposite_pit - 1]
                    if captured > 0:
                        p2_pits[opposite_pit - 1] = 0
                        p1_store += captured + 1
                        p1_pits[current_pit - 1] = 0
        else:
            if current_pit < N:
                current_pit += 1
            elif current_pit == N:
                if current_player == '2':
                    p2_store += 1
                    seeds -= 1
                    if seeds == 0: return p1_pits, p2_pits, p1_store, p2_store, '2'
                side = '1'
                current_pit = 0
                continue
            if current_pit <= N:
                p2_pits[current_pit - 1] += 1
                seeds -= 1
                if seeds == 0 and current_player == '2' and p2_pits[current_pit - 1] == 1:
                    opposite_pit = N - current_pit + 1
                    captured = p1_pits[opposite_pit - 1]
                    if captured > 0:
                        p1_pits[opposite_pit - 1] = 0
                        p2_store += captured + 1
                        p2_pits[current_pit - 1] = 0

    # Switches player based on who's playing.
    next_player = '2' if current_player == '1' else '1'
    return p1_pits, p2_pits, p1_store, p2_store, next_player

# Change boards without affecting who is player 1 and who is player 2.
def apply_PIE(p1_pits, p2_pits, p1_store, p2_store):
    return p2_pits[:], p1_pits[:], p2_store, p1_store

# A check of the termination condition, which is when either player has no seeds left.
def game_is_over(p1_pits, p2_pits):
    return sum(p1_pits) == 0 or sum(p2_pits) == 0

# After game_is_over returns True, take anything still in the pits and add to players' respective stores.
def finalize_game(p1_pits, p2_pits, p1_store, p2_store):
    if sum(p1_pits) == 0:
        p2_store += sum(p2_pits)
        p2_pits = [0]*len(p2_pits)
    elif sum(p2_pits) == 0:
        p1_store += sum(p1_pits)
        p1_pits = [0]*len(p1_pits)
    return p1_pits, p2_pits, p1_store, p2_store

'''
Section: Helper Functions
The following functions are intermediaries between our game logic functions above
and the heuristic functions below.
'''

# Adds any eligible pits to a valid moves list and also PIE if player two is on turn two.
def get_valid_moves(p1_pits, p2_pits, turn, current_player):
    N = len(p1_pits)
    valid_moves = []
    # Invoking PIE is only possible if player two is on turn two.
    if turn == 2 and current_player == '2': valid_moves.append("PIE")
    if current_player == '1':
        for i in range(N):
            if p1_pits[i] > 0: valid_moves.append(i+1)
    else:
        for i in range(N):
            if p2_pits[i] > 0: valid_moves.append(i+1)
    return valid_moves

# Helps the heuristic determine how many seeds can be captured in a given move.
def get_capture_size(p1_pits, p2_pits, p1_store, p2_store, current_player, pit_choice):
    new_p1_pits, new_p2_pits, new_p1_store, new_p2_store, next_player = (
        apply_move(p1_pits, p2_pits, p1_store, p2_store, current_player, pit_choice))
    if current_player == '1': return max(0, (sum(p2_pits) - sum(new_p2_pits)))
    else: return max(0, (sum(p1_pits) - sum(new_p1_pits)))

# Runs a test pit selection and if the next player is still us, we know we got an extra turn.
def get_extra_turn(p1_pits, p2_pits, p1_store, p2_store, current_player, pit_choice):
    new_p1_pits, new_p2_pits, new_p1_store, new_p2_store, next_player = (
        apply_move(p1_pits, p2_pits, p1_store, p2_store, current_player, pit_choice))
    return next_player == current_player

# Checks to see if there is potential for capture, and if there is, attaches a penalty value for our heuristic.
def get_vulnerability(current_pits, opponent_pits):
    vulnerability = 0
    N = len(current_pits)
    for i in range(N):
        if current_pits[i] == 1 and opponent_pits[N - 1 - i] > 0: vulnerability += opponent_pits[N - 1 - i]
    return vulnerability

'''
Section: Heuristic Functions
The following functions assign weights to our hyperparameters,
as well as determine if PIE is a good move or not on turn two.
'''

# This function attaches a score to a move using the hyperparameters and their respective weights.
def heuristic(p1_pits, p2_pits, p1_store, p2_store, current_player, turn):
    # Variables we need.
    extra_move_count = 0
    capture_sum = 0

    # These are the hyperparameters.
    store_difference_weight = 5.0
    pit_difference_weight   = 0.3
    extra_move_weight       = 2.0
    capture_weight          = 4.0
    mobility_weight         = 0.5
    vulnerability_weight    = 3.0

    # Adjusts response based on the current player.
    if current_player == '1':
        store_difference = p1_store - p2_store
        pit_difference = sum(p1_pits) - sum(p2_pits)
        vulnerability = get_vulnerability(p1_pits, p2_pits)
        valid_moves = get_valid_moves(p1_pits, p2_pits, turn, '1')
    else:
        store_difference = p2_store - p1_store
        pit_difference = sum(p2_pits) - sum(p1_pits)
        vulnerability = get_vulnerability(p2_pits, p1_pits)
        valid_moves = get_valid_moves(p1_pits, p2_pits, turn, '2')

    # Degrees of freedom based on how many pits we can choose from (non-zero pits).
    mobility = len([move for move in valid_moves if move != "PIE"])
    for move in valid_moves:
        if move != "PIE":
            if get_extra_turn(p1_pits, p2_pits, p1_store, p2_store, current_player, move): extra_move_count += 1
            capture_sum += get_capture_size(p1_pits, p2_pits, p1_store, p2_store, current_player, move)

    # Evaluation function using hyperparameters and qualities of the board for our minimax algorithm.
    score = (store_difference_weight * store_difference +
             pit_difference_weight   * pit_difference +
             extra_move_weight       * extra_move_count +
             capture_weight          * capture_sum +
             mobility_weight         * mobility -
             vulnerability_weight    * vulnerability)

    return score

# This function attaches a score to invoking PIE to determine whether it's an optimal move.
def evaluate_PIE(p1_pits, p2_pits, p1_store, p2_store, current_player, turn):
    score_without_PIE = heuristic(p1_pits, p2_pits, p1_store, p2_store, current_player, turn)
    if current_player == '2' and turn == 2:
        swapped_p1, swapped_p2, swapped_s1, swapped_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
        score_with_PIE = heuristic(swapped_p1, swapped_p2, swapped_s1, swapped_s2, current_player, turn)
        return max(score_without_PIE, score_with_PIE)
    else: return score_without_PIE

'''
Minimax + Alpha-Beta Pruning Function
The following functions help us prune our game tree strictly to
paths that have value for our program.
'''

# Creating the mechanism for alpha-beta pruning.
def alpha_beta(p1_pits, p2_pits, p1_store, p2_store, turn, current_player,
               alpha, beta, stored_depth, start_time, time_limit, root_player):

    # This will run as long as it has time to.
    if (time.time() - start_time) >= time_limit:
        stored_best_value = evaluate_PIE(p1_pits, p2_pits, p1_store, p2_store, root_player, turn)
        return stored_best_value, None
    if game_is_over(p1_pits, p2_pits) or stored_depth == 0:
        temp_p1_pits, temp_p2_pits, temp_p1_stores, temp_p2_stores = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        stored_best_value = evaluate_PIE(temp_p1_pits, temp_p2_pits, temp_p1_stores, temp_p2_stores, root_player, turn)
        return stored_best_value, None

    # If there are still moves to be made, otherwise check for the termination condition.
    valid_moves = get_valid_moves(p1_pits, p2_pits, turn, current_player)
    if not valid_moves:
        temp_p1_pits, temp_p2_pits, temp_p1_stores, temp_p2_stores = (
            finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store))
        stored_best_value = evaluate_PIE(temp_p1_pits, temp_p2_pits, temp_p1_stores, temp_p2_stores, root_player, turn)
        return stored_best_value, None

    # What our board looks like.
    board_state = (tuple(p1_pits), tuple(p2_pits), p1_store, p2_store, turn, current_player, stored_depth)
    if board_state in transposition_table:
        stored_value, stored_best_move, stored_depth = transposition_table[board_state]
        if stored_depth >= stored_depth: return stored_value, stored_best_move

    # Order moves: try PIE first (if available), then sort by capture potential.
    def move_priority(move):
        if move == "PIE":
            return -1
        else:
            return get_capture_size(p1_pits, p2_pits, p1_store, p2_store, current_player, move)
    valid_moves.sort(key=move_priority, reverse=True)

    # Going down the tree with (-inf, inf).
    stored_best_value = -math.inf
    stored_best_move = None
    for move in valid_moves:
        if move == "PIE":
            new_p1, new_p2, new_s1, new_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
            nxt_player = '1'
            nxt_turn = turn + 1
            stored_best_value, stored_best_move = alpha_beta(new_p1, new_p2, new_s1, new_s2, nxt_turn, nxt_player,
                                                             alpha, beta, stored_depth - 1, start_time, time_limit, root_player)
        else:
            pit = move
            new_p1, new_p2, new_s1, new_s2, nxt_player = apply_move(p1_pits, p2_pits, p1_store, p2_store, current_player, pit)
            nxt_turn = turn if nxt_player == current_player else turn + 1
            stored_best_value, stored_best_move = alpha_beta(new_p1, new_p2, new_s1, new_s2, nxt_turn, nxt_player,
                                                             alpha, beta, stored_depth - 1, start_time, time_limit, root_player)
        if stored_best_value > stored_best_value:
            stored_best_value = stored_best_value
            stored_best_move = move
        alpha = max(alpha, stored_best_value)
        if alpha >= beta: break

    transposition_table[board_state] = (stored_best_value, stored_best_move, stored_depth)
    return stored_best_value, stored_best_move

# Executing the alpha-beta pruning with IDS.
def alpha_beta_search(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player, time_limit=0.9):
    start_time = time.time()
    best_move = None
    max_depth = 50
    for depth in range(1, max_depth + 1):
        if time.time() - start_time >= time_limit: break
        val, move = alpha_beta(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
                               -math.inf, math.inf, depth, start_time, time_limit, cur_player)
        if (time.time() - start_time) < time_limit and move is not None:
            best_move = move
        else: break

    return best_move

'''
Main
The following functions are the brain of the program and are put into a main() function
to avoid scope name shadowing warnings.
'''

# Strictly defined to avoid scope name shadowing warnings which were incredibly frustrating.
def main():
   # Taking input.
    line = sys.stdin.readline().strip()
    if not line: sys.exit(0)
    input_segments = line.split()

    # Formatting input.
    N = int(input_segments[1])
    p1_pits = list(map(int, input_segments[2:2 + N]))
    p2_pits = list(map(int, input_segments[2 + N:2 + 2 * N]))
    p1_store = int(input_segments[2 + 2 * N])
    p2_store = int(input_segments[3 + 2 * N])
    turn = int(input_segments[-2])
    current_player = input_segments[-1]

    # Running program.
    transposition_table.clear()
    best_move = alpha_beta_search(p1_pits, p2_pits, p1_store, p2_store, turn, current_player, time_limit=0.9)
    if not best_move:
        valid_moves = get_valid_moves(p1_pits, p2_pits, turn, current_player)
        best_move = valid_moves[0] if valid_moves else 1
    print(best_move)

# Running main()
if __name__ == "__main__":
    main()
