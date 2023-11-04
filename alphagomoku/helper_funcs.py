import numpy as np

def get_valid_moves(board):
    return [(x, y) for x, y in zip(* np.where(board == 0))]

def get_occupied_locs(board):
    return [(x, y) for x, y in zip(* np.where(board != 0))]

def get_normal_moves(board, dist = 2):
    board_size = board.shape
    occupied = get_occupied_locs(board)
    if not occupied:
        occupied.append((board_size[0] // 2, board_size[1] // 2))
    normal_moves = []
    for move in get_valid_moves(board):
        for loc in occupied:
            if abs(loc[0] - move[0]) <= dist and abs(loc[1] - move[1]) <= dist:
                normal_moves.append(move)
                break
    return normal_moves