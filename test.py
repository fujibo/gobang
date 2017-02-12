import numpy as np
import numba
import multiprocessing as mp
import time
import main
from evaluation import *
# NxN bang
# M moku
# board 0: blank, 1: white, -1: black
N = 7
M = 4
Fsize = N * N

def play(model):

    # parameters
    gamma = 0.9

    board = np.zeros((N, N), dtype=np.int8)
    turn = True

    while True:
        # change black and white
        if not turn:
            board = -board
            actions = np.array(np.where(board == 0))
            r = input()
            r = map(int, r.split(' '))
            board[next(r), next(r)] = 1

            board = -board
            main.dispBoard(board)
            if main.winning(-board):
                return
            turn = not turn

        # can move
        actions = np.array(np.where(board == 0))
        # as feature vectors
        # features = main.getFeatures(board, actions)

        # set algorithm here.
        # r = np.argmax(model.get(features)[:, 0])
        r = main.getMove(board, model, True, 2)

        action = actions[:, r]

        Reward = main.reward(board, action)

        # put
        board[action[0], action[1]] = 1

        # restore black and white
        if not turn:
            board = -board

        main.dispBoard(board)

        # all masses are filled, win
        if Reward != 0:
            return

        # end of this turn
        turn = not turn

if __name__ == '__main__':
    model = main.MyChain()
    serializers.load_npz('./params/10000.model', model)
    play(model=model)
