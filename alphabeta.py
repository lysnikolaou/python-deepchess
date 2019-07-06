import time

import chess

import model
import utils


def eval(board, alpha, beta, player):
    if ((player == chess.WHITE and alpha.turn == chess.WHITE and alpha.is_checkmate())
            or (player == chess.BLACK and beta.turn == chess.BLACK and beta.is_checkmate())):
        return board
    if player == chess.WHITE and model.predict_mlp(utils.bitify(board.fen()), utils.bitify(alpha.fen()))[0] < 0.5:
        return alpha
    elif player == chess.BLACK and model.predict_mlp(utils.bitify(board.fen()), utils.bitify(beta.fen()))[0] < 0.5:
        return beta
    
    return board


def alphabeta(board: chess.Board, depth: int, alpha: chess.Board, beta: chess.Board, player: chess.Color):
    if depth == 0:
        return eval(board, alpha, beta, player)
    
    if player == chess.WHITE:
        for move in board.generate_legal_moves():
            new_board = board.copy()
            new_board.push(move)
            alpha = alphabeta(new_board, depth-1, alpha, beta, chess.BLACK)
            if model.predict_mlp(utils.bitify(beta.fen()), utils.bitify(alpha.fen()))[0] == 1:
                break
        return alpha
    else:
        for move in board.generate_legal_moves():
            new_board = board.copy()
            new_board.push(move)
            beta = alphabeta(new_board, depth-1, alpha, beta, chess.WHITE)
            if model.predict_mlp(utils.bitify(beta.fen()), utils.bitify(alpha.fen()))[0] == 1:
                break
        return beta


def get_optimal_move(board: chess.Board, time_allowed: int):
    depth = 0
    
    alpha = chess.Board('1r3K2/8/5k2/8/8/8/8/8 w - - 0 1')
    beta = chess.Board('2R3k1/8/6K1/8/8/8/8/8 b - - 0 1')
    time_elapsed = float(0)
    while True:
        start = time.time()
        best_board = alphabeta(board, depth, alpha, beta, chess.WHITE)
        end = time.time()
        time_elapsed += end - start
        depth += 1
        if (time_elapsed > time_allowed): break
    return best_board

if __name__ == "__main__":
    print(get_optimal_move(chess.Board(), 3000))
