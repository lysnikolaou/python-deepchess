import csv
import itertools

import numpy as np
import bitstring
import chess
from chess import pgn


PGN_FILENAME = '/home/ubuntu/repos/pychess/games.pgn'


def get_positions(game):
    if len(game.variations) > 0:
        variation = game.variations[0]
    else:
        return []
    while True:
        current_board: chess.Board = variation.board()
        yield current_board.fen()
        if len(variation.variations) > 0:
            variation = variation.variations[0]
        else:
            break


def write_game_row(csv_writer, game):
    for position in get_positions(game):
        label = game.headers['Result']
        csv_writer.writerow([position, label])


def parse_pgn(pgn_file):
    with open('data.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=';')
        csv_writer.writerow(['POSITION', 'RESULT'])
        i = 0
        while i < 100:
            game = pgn.read_game(pgn_file)
            write_game_row(csv_writer, game)
            i += 1


def main():
    pgn_file = open(PGN_FILENAME)
    parse_pgn(pgn_file)
    pgn_file.close()


if __name__ == "__main__":
    main()
