import numpy as np
import numba
import multiprocessing as mp
import time
import main
# NxN bang
# M moku
# board 0: blank, 1: white, -1: black
N = 7
M = 4
Fsize = N*N * (N*N-1) * 2 + N*N

if __name__ == '__main__':

    start = time.time()
    b = np.zeros((7, 7), dtype=np.int8)
    b[3, 3] = 1
    b[3, 4] = 1
    b[3, 5] = 1
    a = np.array(np.where(b == 0))
    for i in range(6):
        for j in range(1, 10):
            w = np.load('./weight/weights{}000_{}.npy'.format(j, i))
            features = main.getFeatures(-b, a)
            print(w.dot(features.transpose()))



    b, res, moved = play(w1, w2)
    dispBoard(b)
    print(moved)
    print("play time", time.time() - start)
