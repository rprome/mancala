#!/usr/bin/env python3

import sys
import math
import time
from datetime import datetime


################################################################################
# Board & Move Logic
################################################################################

def apply_move(p1_pits, p2_pits, p1_store, p2_store, cur_player, chosen_pit):
    """
    Distribute stones from pit 'chosen_pit' (1-based index) on 'cur_player' side.
    Return (new_p1_pits, new_p2_pits, new_p1_store, new_p2_store, next_player).
    Handles capturing, skipping opponent's store, awarding extra move if last stone
    lands in current player's store, etc.
    """
    N = len(p1_pits)
    p1_pits = p1_pits[:]  # copy so we don't mutate caller
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
                # maybe place in p1_store if cur_player is '1'
                if cur_player == '1':
                    p1_store += 1
                    stones -= 1
                    if stones == 0:
                        # last stone in own store => extra turn
                        return (p1_pits, p2_pits, p1_store, p2_store, '1')
                # then switch to side 2, pit 0
                side = '2'
                idx = 0
                continue

            # place a stone
            if idx <= N - 1:
                p1_pits[idx - 1] += 1
                stones -= 1
                # check capture
                if stones == 0 and cur_player == '1' and p1_pits[idx - 1] == 1:
                    opposite_idx = (N - idx + 1)
                    captured = p2_pits[opposite_idx - 1]
                    if captured > 0:
                        p2_pits[opposite_idx - 1] = 0
                        p1_store += (captured + 1)
                        p1_pits[idx - 1] = 0

        else:  # side == '2'
            if idx < N:
                idx += 1
            else:
                # maybe place in p2_store if cur_player == '2'
                if cur_player == '2':
                    p2_store += 1
                    stones -= 1
                    if stones == 0:
                        # last stone in own store => extra turn
                        return (p1_pits, p2_pits, p1_store, p2_store, '2')
                side = '1'
                idx = 0
                continue

            if idx <= N - 1:
                p2_pits[idx - 1] += 1
                stones -= 1
                # check capture
                if stones == 0 and cur_player == '2' and p2_pits[idx - 1] == 1:
                    opposite_idx = (N - idx + 1)
                    captured = p1_pits[opposite_idx - 1]
                    if captured > 0:
                        p1_pits[opposite_idx - 1] = 0
                        p2_store += (captured + 1)
                        p2_pits[idx - 1] = 0

    # If we get here, last stone was NOT in our store => opponent's turn
    next_player = '2' if (cur_player == '1') else '1'
    return (p1_pits, p2_pits, p1_store, p2_store, next_player)


def apply_PIE(p1_pits, p2_pits, p1_store, p2_store):
    """
    Swap sides/stores for the "PIE" move (available to player 2 on turn=2).
    Return (new_p1_pits, new_p2_pits, new_p1_store, new_p2_store).
    """
    return (p2_pits[:], p1_pits[:], p2_store, p1_store)


def game_is_over(p1_pits, p2_pits):
    """
    Return True if all pits on either side are empty.
    """
    return (sum(p1_pits) == 0 or sum(p2_pits) == 0)


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
    return (p1_pits, p2_pits, p1_store, p2_store)


################################################################################
# Advanced Heuristic
################################################################################

def get_valid_moves(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player):
    """
    Collect all valid moves:
      - pit indices (1..N) with >0 stones on cur_player side
      - plus "PIE" if turn=2 and cur_player='2'.
    """
    N = len(p1_pits)
    valid = []

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
    If 'player' sows from 'pit_choice' (1-based) in the *current* position,
    return how many stones would be captured by that move. We'll do a partial
    move simulation using apply_move. If no capture occurs, return 0.
    """
    new_p1, new_p2, new_s1, new_s2, nxt = apply_move(
        p1_pits, p2_pits, p1_store, p2_store, player, pit_choice
    )
    if player == '1':
        # difference in store from before -> after might catch how many seeds ended up in p1_store
        # but that includes normal sowing into store. Let's specifically see how many were captured
        # by seeing how many seeds total are in the store minus how many were in the store originally,
        # ignoring the normal sow. But for clarity, let's do a direct approach:
        # We'll check how many seeds are removed from p2's side if the last stone landed in an empty pit
        # That was the logic in apply_move. We'll just check the difference in p2's pit sum.
        # Simpler: the difference in p1_store might reveal the capture. However, we do not want
        # to count the normal store increment. The normal store increment from sowing is either 1 or 0 if we skip.
        # For simplicity, let's do:
        #   captured_size = (sum of p2_pits) - (sum of new_p2)
        # But that includes all seeds from p2's side. Actually that might also include seeds sowed into p2's side.
        # Let's just check the difference in p1_store from before to after minus 1 if the last stone landed in p1_store.
        # It's simpler to see if the next_player == '1' with an empty pit.
        # However, to keep it straightforward, let's do a direct approach:
        # We'll re-run that logic or we can just keep track from the code above.
        # *Since we've already done the move with apply_move, let's see the net difference in store ignoring standard increments.

        return (new_s1 - p1_store) - (1 if nxt == '1' else 0)
    else:  # player == '2'
        return (new_s2 - p2_store) - (1 if nxt == '2' else 0)


def one_move_earns_extra_turn(p1_pits, p2_pits, p1_store, p2_store, player, pit_choice):
    """
    Returns True if sowing from 'pit_choice' yields an extra move for 'player'
    (i.e., the last stone lands in player's store).
    """
    new_p1, new_p2, new_s1, new_s2, nxt = apply_move(
        p1_pits, p2_pits, p1_store, p2_store, player, pit_choice
    )
    return (nxt == player)


def heuristic(p1_pits, p2_pits, p1_store, p2_store, perspective, turn):
    """
    A more advanced evaluation that considers:
      1) Difference of seeds in the stores (base).
      2) Whether we can get an extra move next turn (prioritize).
      3) Potential to capture (steal) many seeds from opponent.

    We'll:
      - compute base store difference from `perspective`
      - look at the "best" next pit for that perspective in terms of capturing & extra turn
      - combine these into a single numeric score

    You can tweak the weights as desired.
    """
    # Weights for each factor (feel free to adjust)
    W_STORE_DIFF = 1.0
    W_EXTRA_MOVE = 3.0
    W_BIG_STEAL = 2.0

    # 1) Store difference (from perspective)
    if perspective == '1':
        base_score = p1_store - p2_store
    else:
        base_score = p2_store - p1_store

    # 2) Potential next-move analysis: among all valid pits for `perspective`,
    #    find the maximum possible capture plus note if an extra turn is possible.
    valid = get_valid_moves(p1_pits, p2_pits, p1_store, p2_store, turn, perspective)

    max_steal = 0
    can_extra_move = False

    for mv in valid:
        if mv == "PIE":
            # If perspective == '2', then "PIE" is possible. After PIE, the perspective changes to '1'.
            # It's tricky to evaluate capturing from the swapped perspective. We'll do a rough approach:
            new_p1, new_p2, new_s1, new_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
            # after PIE, the next player is '1', turn -> turn+1
            # We won't do a second-level check here. Let's just skip steal for PIE for simplicity,
            # or you could do a mini-check of what '1' could do next, etc.
            # We'll skip it for now:
            pass
        else:
            # It's a pit move
            pit = mv
            # potential stolen seeds
            captured_amount = one_move_capture_size(
                p1_pits, p2_pits, p1_store, p2_store, perspective, pit
            )
            if captured_amount > max_steal:
                max_steal = captured_amount

            # check extra move possibility
            if one_move_earns_extra_turn(p1_pits, p2_pits, p1_store, p2_store, perspective, pit):
                can_extra_move = True

    # Add these to the final evaluation
    extra_move_bonus = W_EXTRA_MOVE if can_extra_move else 0.0
    steal_bonus = W_BIG_STEAL * max_steal

    # Weighted sum
    score = (W_STORE_DIFF * base_score) + extra_move_bonus + steal_bonus
    return score


################################################################################
# Alpha-Beta with new heuristic
################################################################################

def alpha_beta(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
               alpha, beta, depth, start_time, time_limit):
    """
    Standard alpha-beta recursion.  We measure utility from `cur_player`'s perspective.
    Returns (best_value, best_move).
    """
    # Check time
    if (time.time() - start_time) >= time_limit:
        val = heuristic(p1_pits, p2_pits, p1_store, p2_store, cur_player, turn)
        return (val, None)

    # If game over or depth=0, finalize and evaluate
    if game_is_over(p1_pits, p2_pits) or depth == 0:
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = heuristic(tmp1, tmp2, s1, s2, cur_player, turn)
        return (val, None)

    moves = get_valid_moves(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player)
    if not moves:
        # no moves => finalize & evaluate
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = heuristic(tmp1, tmp2, s1, s2, cur_player, turn)
        return (val, None)

    best_val = -math.inf
    best_move = None

    for move in moves:
        if move == "PIE":
            # apply PIE => sides swap
            new_p1, new_p2, new_s1, new_s2 = apply_PIE(p1_pits, p2_pits, p1_store, p2_store)
            # Turn is used, next player is '1', turn -> turn+1
            nxt_player = '1'
            nxt_turn = turn + 1
            val, _ = alpha_beta(new_p1, new_p2, new_s1, new_s2,
                                nxt_turn, nxt_player,
                                alpha, beta, depth - 1,
                                start_time, time_limit)
        else:
            # normal pit move
            pit = move
            new_p1, new_p2, new_s1, new_s2, nxt_player = apply_move(
                p1_pits, p2_pits, p1_store, p2_store, cur_player, pit
            )
            if nxt_player == cur_player:
                nxt_turn = turn
            else:
                nxt_turn = turn + 1

            val, _ = alpha_beta(new_p1, new_p2, new_s1, new_s2,
                                nxt_turn, nxt_player,
                                alpha, beta, depth - 1,
                                start_time, time_limit)

        if val > best_val:
            best_val = val
            best_move = move

        alpha = max(alpha, best_val)
        if alpha >= beta:
            break  # alpha–beta cutoff

    return (best_val, best_move)


def alpha_beta_search(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player, time_limit=0.9):
    """
    Iterative deepening or fixed-depth alpha-beta to find the best move for `cur_player`.
    We'll do up to depth=8 or time-limited, using our advanced heuristic.
    Returns the best move found.
    """
    start_time = time.time()
    best_move = None
    max_depth = 8

    for depth in range(1, max_depth + 1):
        if (time.time() - start_time) >= time_limit:
            break
        val, move = alpha_beta(p1_pits, p2_pits, p1_store, p2_store,
                               turn, cur_player,
                               -math.inf, math.inf, depth,
                               start_time, time_limit)
        # If we still have time, update best_move
        if (time.time() - start_time) < time_limit:
            best_move = move
        else:
            break

    return best_move


################################################################################
# main() - Reads ONE "STATE ..." line, prints ONE move, exits.
################################################################################


# Read exactly one line from stdin
line = sys.stdin.readline().strip()
# Example format:
# STATE 6 4 4 4 4 4 4 4 4 4 4 4 4 0 0 1 1
parts = line.split()
# parse
N = int(parts[1])
p1_pits = list(map(int, parts[2: 2 + N]))
p2_pits = list(map(int, parts[2 + N: 2 + 2 * N]))
p1_store = int(parts[2 + 2 * N])
p2_store = int(parts[3 + 2 * N])
turn = int(parts[-2])
cur_player = parts[-1]  # '1' or '2'

# Find best move with alpha-beta
best_move = alpha_beta_search(
    p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
    time_limit=0.95
)

# If none found (very unlikely), pick a valid move if any
if not best_move:
    valid_moves = get_valid_moves(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player)
    if valid_moves:
        best_move = valid_moves[0]
    else:
        best_move = 1  # fallback

print(best_move)