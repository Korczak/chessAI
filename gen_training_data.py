import os
import numpy as np

import chess
import chess.pgn

from state import State

def save_numpy(patch, X, y):
	np.savez('processed/dataset_{}.npz'.format(patch), np.array(X), np.array(y))

def get_dataset(num_samples=None):
	X,Y = [], []
	iter_to_show = 1
	num_examples = 0
	gn = 0
	values = {'0-1':0, '1-0':1}
	# pgn files in the data folder
	for fn in os.listdir("dataset"):
		print(fn, len(X))
		pgn = open(os.path.join("dataset", fn))
		while 1:

			try:
				game = chess.pgn.read_game(pgn)
			except:
				print('Couldnt load game {}'.format(gn))
				continue
			if game is None:
				break
			res = game.headers['Result']
			if res not in values or res == '1/2-1/2':
				continue
			value = values[res]
			board = game.board()
			for i, move in enumerate(game.mainline_moves()):
				board.push(move)
				if board.turn == True:
					board = board.mirror()
					if value == 0:
						value = 1
					else:
						value = 0
					
				ser = State(board).serialize()
				X.append(ser)
				Y.append(value)
				num_examples += 1
			if len(X) > 100000:
				save_numpy(iter_to_show, X, Y)
				X, Y = [], []
				print("parsing game %d, got %d examples" % (gn, num_examples))
				iter_to_show += 1

			if num_samples is not None and len(X) > num_samples:
				return X,Y
			gn += 1
		X, Y = [], []

	return X,Y


if __name__ == '__main__':
	X, y = get_dataset(10000000)
