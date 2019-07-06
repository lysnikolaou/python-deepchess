import chess

import alphabeta

board = chess.Board()
alpha = chess.Board('2R3k1/8/6K1/8/8/8/8/8 b - - 0 1')
beta = chess.Board('1r3K2/8/5k2/8/8/8/8/8 w - - 0 1')
print(alphabeta.alphabeta(board, 2, alpha, beta, chess.WHITE))