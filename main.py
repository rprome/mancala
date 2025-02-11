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
                if stones == 0 and cur_player == '1' and p1_pits[idx - 1] == 1:
                    # capture
                    opposite_idx = (N - idx + 1)
                    captured = p2_pits[opposite_idx - 1]
                    if captured > 0:
                        p2_pits[opposite_idx - 1] = 0
                        p1_store += (captured + 1)
                        p1_pits[idx - 1] = 0

        else:
            # side == '2'
            if idx < N:
                idx += 1
            else:
                # maybe place in p2_store if cur_player is '2'
                if cur_player == '2':
                    p2_store += 1
                    stones -= 1
                    if stones == 0:
                        return (p1_pits, p2_pits, p1_store, p2_store, '2')
                side = '1'
                idx = 0
                continue

            if idx <= N - 1:
                p2_pits[idx - 1] += 1
                stones -= 1
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
        # add p2's stones to p2_store
        p2_store += sum(p2_pits)
        p2_pits = [0] * len(p2_pits)
    elif sum(p2_pits) == 0:
        # add p1's stones to p1_store
        p1_store += sum(p1_pits)
        p1_pits = [0] * len(p1_pits)
    return (p1_pits, p2_pits, p1_store, p2_store)


################################################################################
# Minimax + Alpha-Beta Search
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


def heuristic(p1_pits, p2_pits, p1_store, p2_store, perspective):
    """
    Simple evaluation: difference in stores from `perspective` player's viewpoint.
    """
    if perspective == '1':
        return p1_store - p2_store
    else:
        return p2_store - p1_store


def alpha_beta(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player,
               alpha, beta, depth, start_time, time_limit):
    """
    Standard alpha-beta recursion.  We measure utility from `cur_player`'s perspective.
    Returns (best_value, best_move).
    """
    # If time is up, return immediate heuristic
    if (time.time() - start_time) >= time_limit:
        val = heuristic(p1_pits, p2_pits, p1_store, p2_store, cur_player)
        return (val, None)

    # If game over or depth=0, finalize and evaluate
    if game_is_over(p1_pits, p2_pits) or depth == 0:
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = heuristic(tmp1, tmp2, s1, s2, cur_player)
        return (val, None)

    moves = get_valid_moves(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player)
    if not moves:
        # no moves? Then finalize and evaluate
        tmp1, tmp2, s1, s2 = finalize_game(p1_pits[:], p2_pits[:], p1_store, p2_store)
        val = heuristic(tmp1, tmp2, s1, s2, cur_player)
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
            # if same player gets another turn, turn doesn't increment
            if nxt_player == cur_player:
                nxt_turn = turn
            else:
                nxt_turn = turn + 1

            # The perspective for the next call is *nxt_player* (the new current player).
            # So the alpha-beta is always from the vantage of the node's current player
            # wanting to maximize the store difference from its perspective.
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
    Reads from perspective = cur_player. We'll do up to, say, depth=8 or time-limited.
    Returns the best move found.
    """
    start_time = time.time()
    best_move = None
    best_val = -math.inf
    max_depth = 8  # adjust if you want

    for depth in range(1, max_depth + 1):
        if (time.time() - start_time) >= time_limit:
            break
        val, move = alpha_beta(p1_pits, p2_pits, p1_store, p2_store,
                               turn, cur_player,
                               -math.inf, math.inf, depth,
                               start_time, time_limit)
        # If we haven't timed out, accept this best move
        if (time.time() - start_time) < time_limit:
            best_val = val
            best_move = move
        else:
            break

    return best_move


################################################################################
# main() - Reads ONE "STATE ..." line, prints ONE move, exits.
################################################################################

def main():
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
    best_move = alpha_beta_search(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player, time_limit=0.95)

    # In case we got nothing, pick a valid move if any
    if not best_move:
        valid_moves = get_valid_moves(p1_pits, p2_pits, p1_store, p2_store, turn, cur_player)
        if valid_moves:
            best_move = valid_moves[0]
        else:
            # Should never happen unless the game is basically over, but let's fail gracefully
            best_move = 1  # arbitrary fallback

    print(best_move)


if __name__ == "__main__":
    main()
