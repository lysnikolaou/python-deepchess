import itertools

import numpy as np
import chess
import bitstring


def bitify(fen: str):
    board = chess.Board(fen)
    result = np.zeros(768+5)
    for idx, element in enumerate(itertools.product(chess.PIECE_TYPES, chess.COLORS)):
        piece = element[0]
        color = element[1]
        piece_mask = board.pieces_mask(piece, color)
        bitarray = bitstring.BitArray(uint=piece_mask, length=64)
        result[idx*64:(idx+1)*64] = [int(i) for i in list(bitarray)]

    result[768] = int(board.has_kingside_castling_rights(chess.WHITE))
    result[769] = int(board.has_queenside_castling_rights(chess.WHITE))
    result[770] = int(board.has_kingside_castling_rights(chess.BLACK))
    result[771] = int(board.has_queenside_castling_rights(chess.BLACK))
    result[772] = int(board.turn)
    return [int(i) for i in result.tolist()]
