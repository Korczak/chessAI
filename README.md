# chessAI

Implementation of Neural Network for Chess.

Used libraries:
- chess
- Keras
- Matplotlib
- Numpy
- PySimpleGUI for Graphics Visualization



Neural Networks predicts which player wins actual Board.
Algorithm calculate prediction for every possible move and choose with the highest prediction of winning.
Board is serialized as (Name: value for white/ value for black): 
- King: 100/-100
- Queen: 9/-9
- Rook: 5/-5
- Bishop: 3/-3
- Knight: 3/-3
- Pawn: 1/-1

