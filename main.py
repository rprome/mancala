import sys
import math
import time

# Global variable for search depth reporting
def apply_move(p1_pits, p2_pits, p1_store, p2_store, current_player, chosen_pit):
    N = len(p1_pits)
    p1_pits = p1_pits[:]
    p2_pits = p2_pits[:]
    if current_player == '1':
        seeds = p1_pits[chosen_pit - 1]
        p1_pits[chosen_pit - 1] = 0
    else:
        seeds = p2_pits[chosen_pit - 1]
        p2_pits[chosen_pit - 1] = 0
    current_pit = chosen_pit
    side = current_player

    while seeds > 0:
        if side == '1':
            if current_pit < N:
                current_pit += 1
            elif current_pit == N:
                if current_player == '1':
                    p1_store += 1
                    seeds -= 1
                    if seeds == 0:
                        return p1_pits, p2_pits, p1_store, p2_store, '1'
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
                    if seeds == 0:
                        return p1_pits, p2_pits, p1_store, p2_store, '2'
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

    next_player = '2' if current_player == '1' else '1'
    return p1_pits, p2_pits, p1_store, p2_store, next_player

def apply_PIE(p1_pits, p2_pits, p1_store, p2_store):
    return p2_pits[:], p1_pits[:], p2_store, p1_store

def game_is_over(p1_pits, p2_pits):
    return sum(p1_pits) == 0 or sum(p2_pits) == 0

def finalize_game(p1_pits, p2_pits, p1_store, p2_store):
    if sum(p1_pits) == 0:
        p2_store += sum(p2_pits)
        p2_pits = [0]*len(p2_pits)
    elif sum(p2_pits) == 0:
        p1_store += sum(p1_pits)
        p1_pits = [0]*len(p1_pits)
    return p1_pits, p2_pits, p1_store, p2_store

'''
Helper Functions
'''

def get_valid_moves(p1_pits, p2_pits, turn, current_player):
    N = len(p1_pits)
    valid = []
    if turn == 2 and current_player == '2':
        valid.append("PIE")
    if current_player == '1':
        for i in range(N):
            if p1_pits[i] > 0:
                valid.append(i+1)
    else:
        for i in range(N):
            if p2_pits[i] > 0:
                valid.append(i+1)
    return valid

def one_move_capture_size(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice):
    new_p1, new_p2, new_s1, new_s2, _ = apply_move(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice)
    if player == '1':
        return max(0, (sum(p2_pits) - sum(new_p2)))
    else:
        return max(0, (sum(p1_pits) - sum(new_p1)))

def one_move_earns_extra_turn(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice):
    _, _, _, _, next_player = apply_move(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice)
    return next_player == player

def vulnerability(pits, opp_pits):
    """Returns a penalty value for pits that are vulnerable to capture.
       For each pit on our side with exactly one seed and an opponent’s opposite pit with seeds,
       we add a penalty equal to the number of seeds in the opponent's pit.
    """
    penalty = 0
    N = len(pits)
    for i in range(N):
        if pits[i] == 1 and opp_pits[N - 1 - i] > 0:
            penalty += opp_pits[N - 1 - i]
    return penalty

'''
Heuristic
'''

def evaluate_position(p1_pits, p2_pits, p1_store, p2_store, perspective, turn):
    """
    Enhanced evaluation with additional defensive (vulnerability) factor.
    Factors (with example weights):
      - Store difference (W_STORE_DIFF = 5.0)
      - Pit seeds difference (W_PIT_DIFF = 0.3)
      - Extra move potential (W_EXTRA_MOVE = 2.0)
      - Capture potential (W_CAPTURE = 4.0)
      - Mobility (W_MOBILITY = 0.5)
      - Vulnerability penalty (W_VULN = 3.0) [subtracted]
    """
    W_STORE_DIFF = 5.0
    W_PIT_DIFF   = 0.3
    W_EXTRA_MOVE = 2.0
    W_CAPTURE    = 4.0
    W_MOBILITY   = 0.5
    W_VULN       = 3.0  # penalty weight

    if perspective == '1':
        store_diff = p1_store - p2_store
        pit_diff = sum(p1_pits) - sum(p2_pits)
        vuln_penalty = vulnerability(p1_pits, p2_pits)
        valid_moves = get_valid_moves(p1_pits, p2_pits, turn, '1')
    else:
        store_diff = p2_store - p1_store
        pit_diff = sum(p2_pits) - sum(p1_pits)
        vuln_penalty = vulnerability(p2_pits, p1_pits)
        valid_moves = get_valid_moves(p1_pits, p2_pits, turn, '2')

    extra_move_count = 0
    capture_sum = 0
    mobility = len([m for m in valid_moves if m != "PIE"])
    for move in valid_moves:
        if move != "PIE":
            if one_move_earns_extra_turn(p1_pits, p2_pits, p1_store, p2_store, perspective, move):
                extra_move_count += 1
            capture_sum += one_move_capture_size(p1_pits, p2_pits, p1_store, p2_store, perspective, move)

    score = (W_STORE_DIFF * store_diff +
             W_PIT_DIFF   * pit_diff +
             W_EXTRA_MOVE * extra_move_count +
             W_CAPTURE    * capture_sum +
             W_MOBILITY   * mobility -
             W_VULN       * vuln_penalty)
    return score

def evaluate_PIE(p1_pits, p2_pits, p1_store, p2_store, current_player, turn):
    score_without_PIE = evaluate_position(p1_pits, p2_pits, p1_store, p2_store, current_player, turn)
    if current_player == '2' and turn == 2:
        swapped_p1, swapped_p2, swapped_s1, swapped_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
        score_with_PIE = evaluate_position(swapped_p1, swapped_p2, swapped_s1, swapped_s2, current_player, turn)
        return max(score_without_PIE, score_with_PIE)
    else:
        return score_without_PIE

'''
Alpha-Beta Pruning
'''

transposition_table = {}

def alpha_beta(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
               alpha, beta, depth, start_time, time_limit, root_player):
    if (time.time() - start_time) >= time_limit:
        val = evaluate_PIE(p1_pits, p2_pits, p1_store, p2_store, root_player, turn)
        return val, None

    if game_is_over(p1_pits, p2_pits) or depth == 0:
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = evaluate_PIE(tmp1, tmp2, s1, s2, root_player, turn)
        return val, None

    moves = get_valid_moves(p1_pits, p2_pits, turn, cur_player)
    if not moves:
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = evaluate_PIE(tmp1, tmp2, s1, s2, root_player, turn)
        return val, None

    board_key = (tuple(p1_pits), tuple(p2_pits), p1_store, p2_store, turn, cur_player, depth)
    if board_key in transposition_table:
        cached_val, cached_move, cached_depth = transposition_table[board_key]
        if cached_depth >= depth:
            return cached_val, cached_move

    # Order moves: try PIE first (if available), then sort by capture potential.
    def move_priority(m):
        if m == "PIE":
            return -1
        else:
            return one_move_capture_size(p1_pits, p2_pits, p1_store, p2_store, cur_player, m)
    moves.sort(key=move_priority, reverse=True)

    best_val = -math.inf
    best_move = None
    for move in moves:
        if move == "PIE":
            new_p1, new_p2, new_s1, new_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
            nxt_player = '1'
            nxt_turn = turn + 1
            val, _ = alpha_beta(new_p1, new_p2, new_s1, new_s2, nxt_turn, nxt_player,
                                alpha, beta, depth - 1, start_time, time_limit, root_player)
        else:
            pit = move
            new_p1, new_p2, new_s1, new_s2, nxt_player = apply_move(p1_pits, p2_pits, p1_store, p2_store, cur_player, pit)
            nxt_turn = turn if nxt_player == cur_player else turn + 1
            val, _ = alpha_beta(new_p1, new_p2, new_s1, new_s2, nxt_turn, nxt_player,
                                alpha, beta, depth - 1, start_time, time_limit, root_player)
        if val > best_val:
            best_val = val
            best_move = move
        alpha = max(alpha, best_val)
        if alpha >= beta:
            break

    transposition_table[board_key] = (best_val, best_move, depth)
    return best_val, best_move

def alpha_beta_search(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player, time_limit=0.9):
    start_time = time.time()
    best_move = None
    max_depth = 50  # try to search as deeply as time permits
    global global_depth
    for depth in range(1, max_depth + 1):
        if time.time() - start_time >= time_limit:
            break
        global_depth = depth
        val, move = alpha_beta(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
                               -math.inf, math.inf, depth, start_time, time_limit, cur_player)
        if (time.time() - start_time) < time_limit and move is not None:
            best_move = move
        else:
            break
    print(f"Max depth reached: {global_depth}", file=sys.stderr)
    return best_move

#####################################
# Main Function
#####################################

def main():
    line = sys.stdin.readline().strip()
    if not line:
        sys.exit(0)
    parts = line.split()
    N = int(parts[1])
    p1_pits = list(map(int, parts[2:2+N]))
    p2_pits = list(map(int, parts[2+N:2+2*N]))
    p1_store = int(parts[2+2*N])
    p2_store = int(parts[3+2*N])
    turn = int(parts[-2])
    cur_player = parts[-1]  # '1' or '2'
    transposition_table.clear()
    best_move = alpha_beta_search(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player, time_limit=0.9)
    if not best_move:
        valid_moves = get_valid_moves(p1_pits, p2_pits, turn, cur_player)
        best_move = valid_moves[0] if valid_moves else 1
    print(best_move)

if __name__ == "__main__":
    main()