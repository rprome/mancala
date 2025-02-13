import sys
import math
import time

################################################################################
# Board & Move Logic
################################################################################

def apply_move(p1_pits, p2_pits, p1_store, p2_store, cur_player, chosen_pit):
    """
    Distribute stones from pit 'chosen_pit' (1-based index) on 'cur_player' side.
    Return (new_p1_pits, new_p2_pits, new_p1_store, new_p2_store, next_player).
    Handles capturing, skipping opponent's store, awarding extra move, etc.
    """
    N = len(p1_pits)
    p1_pits = p1_pits[:]
    p2_pits = p2_pits[:]

    if cur_player == '1':
        stones = p1_pits[chosen_pit - 1]
        p1_pits[chosen_pit - 1] = 0
    else:
        stones = p2_pits[chosen_pit - 1]
        p2_pits[chosen_pit - 1] = 0

    idx = chosen_pit
    side = cur_player  # '1' or '2'

    while stones > 0:
        if side == '1':
            # move to next pit or store
            if idx < N:
                idx += 1
            else:
                # place in p1_store if cur_player is '1'
                if cur_player == '1':
                    p1_store += 1
                    stones -= 1
                    if stones == 0:
                        # last stone in own store => extra turn
                        return p1_pits, p2_pits, p1_store, p2_store, '1'
                # then switch to side 2, pit 0
                side = '2'
                idx = 0
                continue

            # place a stone
            if idx <= N:
                p1_pits[idx - 1] += 1
                stones -= 1
                # check capture if last stone lands in an empty pit on your side
                if stones == 0 and cur_player == '1' and p1_pits[idx - 1] == 1:
                    opposite_idx = N - idx + 1
                    captured = p2_pits[opposite_idx - 1]
                    if captured > 0:
                        p2_pits[opposite_idx - 1] = 0
                        p1_store += (captured + 1)
                        p1_pits[idx - 1] = 0

        else:  # side == '2'
            if idx < N:
                idx += 1
            else:
                # place in p2_store if cur_player == '2'
                if cur_player == '2':
                    p2_store += 1
                    stones -= 1
                    if stones == 0:
                        # last stone in own store => extra turn
                        return p1_pits, p2_pits, p1_store, p2_store, '2'
                side = '1'
                idx = 0
                continue

            if idx <= N:
                p2_pits[idx - 1] += 1
                stones -= 1
                # check capture
                if stones == 0 and cur_player == '2' and p2_pits[idx - 1] == 1:
                    opposite_idx = N - idx + 1
                    captured = p1_pits[opposite_idx - 1]
                    if captured > 0:
                        p1_pits[opposite_idx - 1] = 0
                        p2_store += (captured + 1)
                        p2_pits[idx - 1] = 0

    # If we get here, last stone was NOT in our store => opponent's turn
    next_player = '2' if (cur_player == '1') else '1'
    return p1_pits, p2_pits, p1_store, p2_store, next_player

def apply_move_with_capture_info(p1_pits, p2_pits, p1_store, p2_store, cur_player, chosen_pit):
    """
    Distribute stones from pit 'chosen_pit' on 'cur_player' side.
    Return (new_p1_pits, new_p2_pits, new_p1_store, new_p2_store, next_player, captured_count).
    """
    N = len(p1_pits)
    p1_pits = p1_pits[:]
    p2_pits = p2_pits[:]

    if cur_player == '1':
        stones = p1_pits[chosen_pit - 1]
        p1_pits[chosen_pit - 1] = 0
    else:
        stones = p2_pits[chosen_pit - 1]
        p2_pits[chosen_pit - 1] = 0

    idx = chosen_pit
    side = cur_player  # '1' or '2'
    captured_count = 0  # new variable to track exactly how many stones were captured

    while stones > 0:
        if side == '1':
            if idx < N:
                idx += 1
            else:
                # place in p1 store if it's player 1's turn
                if cur_player == '1':
                    p1_store += 1
                    stones -= 1
                    if stones == 0:
                        # last stone in own store => extra turn
                        return p1_pits, p2_pits, p1_store, p2_store, '1', captured_count
                # switch to side 2, pit 0
                side = '2'
                idx = 0
                continue

            # place stone in p1 pit
            p1_pits[idx - 1] += 1
            stones -= 1

            # check capture
            if stones == 0 and cur_player == '1' and p1_pits[idx - 1] == 1:
                opp_idx = N - (idx - 1)  # or N - idx + 1
                captured = p2_pits[opp_idx - 1]
                if captured > 0:
                    p2_pits[opp_idx - 1] = 0
                    p1_store += (captured + 1)
                    captured_count = captured + 1  # <--- record exactly how many were taken
                    p1_pits[idx - 1] = 0
        else:
            # side == '2'
            if idx < N:
                idx += 1
            else:
                # place in p2 store if it's player 2's turn
                if cur_player == '2':
                    p2_store += 1
                    stones -= 1
                    if stones == 0:
                        # last stone in own store => extra turn
                        return p1_pits, p2_pits, p2_store, p2_store, '2', captured_count
                side = '1'
                idx = 0
                continue

            # place stone in p2 pit
            p2_pits[idx - 1] += 1
            stones -= 1

            # check capture
            if stones == 0 and cur_player == '2' and p2_pits[idx - 1] == 1:
                opp_idx = N - (idx - 1)
                captured = p1_pits[opp_idx - 1]
                if captured > 0:
                    p1_pits[opp_idx - 1] = 0
                    p2_store += (captured + 1)
                    captured_count = captured + 1
                    p2_pits[idx - 1] = 0

    # If we get here, the last stone did not land in our store => next player's turn
    next_player = '2' if (cur_player == '1') else '1'
    return p1_pits, p2_pits, p1_store, p2_store, next_player, captured_count


def apply_PIE(p1_pits, p2_pits, p1_store, p2_store):
    """
    Swap sides/stores for the "PIE" move (available to player 2 on turn=2).
    Return (new_p1_pits, new_p2_pits, new_p1_store, new_p2_store).
    """
    return p2_pits[:], p1_pits[:], p2_store, p1_store


def game_is_over(p1_pits, p2_pits):
    """
    Return True if all pits on either side are empty.
    """
    return sum(p1_pits) == 0 or sum(p2_pits) == 0


def finalize_game(p1_pits, p2_pits, p1_store, p2_store):
    """
    If one side is empty, move all stones from the non-empty side to its store.
    Return the final (p1_pits, p2_pits, p1_store, p2_store).
    """
    if sum(p1_pits) == 0:
        p2_store += sum(p2_pits)
        p2_pits = [0] * len(p2_pits)
    elif sum(p2_pits) == 0:
        p1_store += sum(p1_pits)
        p1_pits = [0] * len(p1_pits)
    return p1_pits, p2_pits, p1_store, p2_store


################################################################################
# Heuristic & Helpers
################################################################################

def get_valid_moves(p1_pits, p2_pits, turn, cur_player):
    """
    Collect all valid moves:
      - pit indices (1.N) with >0 stones on cur_player's side
      - plus "PIE" if turn=2 and cur_player='2'.
    """
    N = len(p1_pits)
    valid = []

    # PIE rule: only available on turn=2 for player 2
    if turn == 2 and cur_player == '2':
        valid.append("PIE")

    if cur_player == '1':
        for i in range(N):
            if p1_pits[i] > 0:
                valid.append(i + 1)
    else:
        for i in range(N):
            if p2_pits[i] > 0:
                valid.append(i + 1)

    return valid


def one_move_capture_size(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice):
    """
    Return how many stones would be captured by 'player' if they sow from 'pit_choice'.
    """
    # We only care about the captured_count from the single move.
    (_new_p1, _new_p2, _new_s1, _new_s2, _nxt, captured_count) = apply_move_with_capture_info(
        p1_pits, p2_pits, p1_store, p2_store, player, pit_choice
    )
    return captured_count


def one_move_earns_extra_turn(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice):
    """
    Returns True if sowing from 'pit_choice' yields an extra move for 'player'
    (last stone lands in player's store).
    """
    new_p1, new_p2, new_s1, new_s2, nxt = apply_move(
        p1_pits, p2_pits, p1_store, p2_store, player, pit_choice
    )
    return nxt == player


def heuristic(p1_pits, p2_pits, p1_store, p2_store, perspective, turn):
    """
    Evaluate the board from the viewpoint of `perspective` ( '1' or '2' ).
    Incorporates the possibility that if `perspective` == '2' and turn == 2,
    PIE might be beneficial.
    """

    # 1) Evaluate the position normally (no PIE).
    score_no_pie = evaluate_position(p1_pits, p2_pits, p1_store, p2_store, perspective, turn)

    # 2) If we can do PIE (meaning perspective=2 and turn=2), evaluate that scenario too.
    if perspective == '2' and turn == 2:
        swapped_p1, swapped_p2, swapped_s1, swapped_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
        score_with_pie = evaluate_position(swapped_p1, swapped_p2, swapped_s1, swapped_s2, perspective, turn)
        # Take whichever is better from Player 2’s viewpoint
        return max(score_no_pie, score_with_pie)
    else:
        return score_no_pie


def evaluate_position(p1_pits, p2_pits, p1_store, p2_store, perspective, turn):
    """
    Your original scoring logic that factors in store difference, side difference,
    potential captures, etc. (without worrying about PIE).
    """
    # Example: your current logic
    W_STORE_DIFF = 1.0
    W_EXTRA_MOVE = 3.0
    W_BIG_STEAL  = 1.0
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
    extra_move_bonus = 0.0
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
        val = heuristic(p1_pits, p2_pits, p1_store, p2_store, root_player, turn)
        return val, None

    # If game over or depth=0, finalize & evaluate
    if game_is_over(p1_pits, p2_pits) or depth == 0:
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = heuristic(tmp1, tmp2, s1, s2, root_player, turn)
        return val, None

    moves = get_valid_moves(p1_pits, p2_pits, turn, cur_player)
    if not moves:
        # no moves => finalize & evaluate
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = heuristic(tmp1, tmp2, s1, s2, root_player, turn)
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


def alpha_beta_search(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
                      time_limit=0.98):
    """
    Iterative deepening alpha-beta to find the best move for `cur_player`.
    We define root_player = cur_player so we evaluate from that perspective.
    """
    start_time = time.time()
    best_move = None
    max_depth = 12  # Increase as desired

    # We'll do iterative deepening from depth=1.max_depth
    for depth in range(1, max_depth + 1):
        if (time.time() - start_time) >= time_limit:
            break
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

    return best_move


################################################################################
# main() - Reads one line "STATE ...", prints one move, then exits
################################################################################

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
        time_limit=0.95
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
