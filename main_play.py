import numpy as np
import numba
import multiprocessing as mp
import time
from main import *
# NxN bang
# M moku
# board 0: blank, 1: white, -1: black

@numba.jit(numba.i8(numba.i1[:, :], numba.f8[:], numba.b1, numba.i8))
def getMove(board, weights, flag, depth):
    'flag: if this function is for idx or for score'
    if depth == 1:
        actions = np.array(np.where(board == 0))
        features = getFeatures(board, actions)
        if flag:
            return np.argmax(weights.dot(features.transpose()))
        else:
            return -np.max(weights.dot(features.transpose()))
    else:
        actions = np.array(np.where(board == 0))
        score = np.zeros(actions.shape[1])
        for i in range(actions.shape[1]):
            nextboard = board.copy()
            nextboard[actions[0, i], actions[1, i]] = 1
            nextboard = -nextboard
            score[i] = getMove(nextboard, weights, False, depth-1)
        else:
            if flag:
                return np.argmax(score)
            else:
                return np.max(score)


def play(weights1, weights2):
    board = np.zeros((N, N), dtype=np.int8)
    moved = []
    turn = True

    while True:
        # change black and white
        if not turn:
            board = -board

        # can move
        actions = np.array(np.where(board == 0))

        if turn:
            r = getMove(board, weights1, True, depth=2)
        else:
            r = getMove(board, weights2, True, depth=2)

        # put
        moved.append(actions[:, r])
        board[actions[0][r], actions[1][r]] = 1

        Reward = 0
        # winning state
        win = winning(board.flatten())
        if win:
            if turn:
                print("white")
                Reward = 1
            else:
                print("black")
                Reward = -1

        # restore black and white
        if not turn:
            board = -board

        # print(actions[0][r], actions[1][r])
        dispBoard(board)



        # end of the game
        if Reward != 0:
            return (board, Reward, moved)

        # all masses are filled.
        elif (board != 0).all():
            print("draw")
            return (board, Reward, moved)

        # end of this turn
        turn = not turn

if __name__ == '__main__':

    start = time.time()
    w1 = np.load('./weight/weights1000_3.npy')
    w2 = np.load('./weight/weights1000_4.npy')

    b, res, moved = play(w1, w2)
    dispBoard(b)
    print(moved)
    print("play time", time.time() - start)
