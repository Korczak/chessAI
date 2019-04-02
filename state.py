import chess
import chess.pgn
import numpy as np

piece_value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 100, \
	   						"p": -1, "n":-3, "b":-3, "r":-5, "q":-9, "k": -100}


def serialize_game(board):
	serialized_board = np.zeros((8, 8), dtype=np.int64)
	for row in range(0, 8):
		for col in range(0, 8):
			piece = board.piece_at(row*8+col)
			if piece == None:
				serialized_board[row][col] = 0
			else:
				serialized_board[row][col] = piece_value[piece.symbol()]
			
	return serialized_board