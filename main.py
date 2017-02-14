import numpy as np
import numba
import multiprocessing as mp
import time
from evaluation import *

# NxN bang
# M moku
# board 0: blank, 1: white, -1: black
N = 7
M = 4
Fsize = N * N * 2


@numba.jit(numba.b1(numba.i1[:]))
def winning(board):
    tf = False

    board = board.reshape(N, N)
    for i in range(N):
        for j in range(N - M + 1):
            # white
            if (board[i, j:j + M] == 1).all():
                tf = True
                # print("row i, j", i, j)
                break
            elif (board[j:j + M, i] == 1).all():
                tf = True
                # print("col i, j", i, j)
                break
    if tf:
        return True

    for i in range(N - M + 1):
        for j in range(N - M + 1):
            tmp = board[i:i + M, j:j + M]
            if (tmp.diagonal() == 1).all():
                tf = True
                # print("diag1 i, j", i, j)
                break

            elif (np.rot90(tmp).diagonal() == 1).all():
                tf = True
                # print("diag2 i, j", i, j)
                break
    return tf


@numba.jit(numba.f4(numba.i1[:, :], numba.i8[:, :]))
def reward(board, action):
    tmp = board.copy()
    tmp[action[0], action[1]] = 1
    reward = 0
    # winning state
    if winning(tmp.flatten()):
        reward = 1.0
        # reward = 4.0 + np.sum(tmp == 0) // N
    elif (tmp != 0).all():
        reward = -0.1
    return reward


@numba.jit(numba.f4[:](numba.i1[:, :], numba.i8[:, :]))
def getFeature(board, action):
    'board: board now, action: one action'

    tmp = board.copy()
    tmp[action[0], action[1]] = 1

    tmp = tmp.flatten()
    feature = tmp.astype(np.float32)
    return feature


@numba.jit(numba.f4[:, :](numba.i1[:, :], numba.i8[:, :]))
def getFeatures(board, actions):
    'board: board now, actions: can put there'
    # use next board(after-an-action) state  as parameters
    Features = board.flatten().reshape(1, board.size).repeat(
        actions.shape[1], axis=0)
    for i in range(actions.shape[1]):
        Features[i, actions[0, i] + actions[1, i] * N] = 1

    Features = np.hstack((Features == 1, Features == -1)).astype(np.float32)
    # Features = np.zeros((actions.shape[1], Fsize), dtype=np.float32)
    # for i in range(actions.shape[1]):
    #     Features[i, :] = getFeature(board, actions[:, i])
    return Features

# @numba.jit(numba.f8[:](numba.f8[:]))


def game(model, eps=0.10):

    xs = []
    nxs = []
    ys = []
    # parameters
    gamma = 0.9
    epsilon = eps

    board = np.zeros((N, N), dtype=np.int8)
    turn = True

    while True:
        # change black and white
        if not turn:
            board = -board

        # can move
        actions = np.array(np.where(board == 0))
        # as feature vectors
        features = getFeatures(board, actions)

        # set algorithm here.

        # epsilon-greedy
        if np.random.rand() < epsilon:
            r = np.random.randint(actions[0].size)
        else:
            # actions[0] x 1
            r = np.argmax(model.get(features)[:, 0])
            # r = np.argmax(weights.dot(features.transpose()))

        action = actions[:, r]
        feature = features[r, :]

        Reward = reward(board, action)

        # update weights

        # all masses are filled, win
        if Reward != 0:

            xs.append(feature)
            nxs.append(board.flatten())
            ys.append(Reward)

            return (xs, nxs, ys)

        # else
        else:
            nextboard = board.copy()
            nextboard[action[0], action[1]] = 1
            # can move
            nextboard = -nextboard
            # nextactions = np.array(np.where(nextboard == 0))
            # # set algorithm here.
            # nextfeatures = getFeatures(nextboard, nextactions)

            xs.append(feature)
            nxs.append(nextboard.flatten())

            # この内の最大となるyを選択したい
            # y = -gamma * np.max(model.get(nextfeatures)[:, 0])
            ys.append(Reward)

        # put
        board[action[0], action[1]] = 1

        # restore black and white
        if not turn:
            board = -board

        # end of this turn
        turn = not turn

def dispBoard(board):
    'display board'
    print("")
    for i in range(N):
        for j in range(N):
            if board[i, j] == 1:
                print("O", end="")
            elif board[i, j] == -1:
                print("X", end="")
            else:
                print(".", end="")
        else:
            print("")


def main(queue, pid):
    model = MyChain()
    # serializers.load_npz('./params_1/99.model', model)
    optimizer = optimizers.Adam()
    optimizer.setup(model)


    x_data = []
    nx_data = []
    y_data = []
    data_size = 0
    # make data

    # init
    stime = time.time()
    for i in range(1, 21):
        xs, nxs, ys = game(model, eps=0.1)
        # xs, ys = game(model)
        num = len(ys)

        x_data += xs
        nx_data += nxs
        y_data += ys
        data_size += num

    print('end of games. takes', time.time() - stime, 'sec')

    gamma = 0.9
    # learning
    for i in range(1000):
        xs, nxs, ys = game(model, eps=0.1)
        # xs, ys = game(model)
        num = len(ys)

        x_data += xs
        nx_data += nxs
        y_data += ys
        data_size += num

        # Fsize x gamesize
        x_data_ = np.array(x_data, dtype=np.float32).reshape(data_size, Fsize)
        nx_data_ = np.array(nx_data, dtype=np.float32).reshape(data_size, N*N)
        # gamesize
        y_data_ = np.array(y_data, dtype=np.float32).reshape(data_size, 1)

        print('sampling')
        sample_t = time.time()
        # 100 boards sampling
        idxes = np.random.permutation(data_size)
        # random sampling
        x = x_data_[idxes[0:100], :]
        nx = nx_data_[idxes[0:100], :]
        y = y_data_[idxes[0:100], :]

        for k in range(100):
            if  y[k, 0] == 0:
                tmp = nx[k, :].reshape(N, N)
                nextactions = np.array(np.where(tmp == 0))
                nextfeatures = getFeatures(tmp, nextactions)
                y[k, 0] = -gamma * np.max(model.get(nextfeatures)[:, 0])

        print('sampling end.', time.time() - sample_t, 'sec')


        losses = []
        for j in range(1, 1001):

            # a x 49
            x_ = Variable(x)
            # a x 1
            y_ = Variable(y)

            model.cleargrads()
            loss = model(x_, y_)
            loss.backward()
            optimizer.update()

            losses.append(loss.data)

            if j % 50 == 0:
                plt.plot(losses, 'b')
                plt.yscale('log')
                plt.pause(1e-12)

            if j % 500 == 0:
                test(model)
        else:
            plt.clf()

        # plt.savefig('./params_1/fig{}.png'.format(i))
        # plt.clf()
        print(i, 'end of learning')
        serializers.save_npz('./params_1/{}_.model'.format(k), model)


    queue.put(1)
    return

@numba.jit(numba.i8(numba.i1[:, :], numba.f8[:], numba.b1, numba.i8))
def getMove(board, model, flag, depth):
    '''board, model, flag, depth
    flag: if this function is for idx or for score'''
    if depth == 1:
        actions = np.array(np.where(board == 0))
        features = getFeatures(board, actions)
        if flag:
            return np.argmax(model.get(features)[:, 0])
        else:
            return -np.max(model.get(features)[:, 0])
    else:
        actions = np.array(np.where(board == 0))
        score = np.zeros(actions.shape[1])
        for i in range(actions.shape[1]):
            nextboard = board.copy()
            nextboard[actions[0, i], actions[1, i]] = 1
            nextboard = -nextboard
            score[i] = getMove(nextboard, model, False, depth-1)
        else:
            if flag:
                return np.argmax(score)
            else:
                return np.max(score)

def test(model):
    start = time.time()
    b = np.zeros((N, N), dtype=np.int8)
    b[3, 2] = 1
    b[3, 3] = 1
    b[3, 4] = 1
    a = np.array(np.where(b == 0))
    fs = getFeatures(b, a)
    score = model.get(fs)[:, 0]
    print(score)
    idx = np.argmax(score)
    # 22, 23 are desirable
    print(idx, "<- idx, ", score[idx], "<- value")
    # print(a[:, idx])
    print(time.time() - start, "sec for all")


if __name__ == '__main__':

    queue = mp.Queue()
    testSize = 1
    pc = 0  # work as program counter
    start = time.time()
    if testSize == 1:
        main(queue, 0)
        pc += 1
    else:
        # ps = [mp.Process(target=main, args=(queue, np.random.rand(1, Fsize)/10, i)) for i in range(testSize)]
        ps = [mp.Process(target=main, args=(queue, i)) for i in range(testSize)]

        while pc < min(mp.cpu_count(), testSize):
            ps[pc].start()
            pc += 1

    result = []
    for i in range(testSize):
        result.append(queue.get())
        if pc < testSize:
            ps[pc].start()
            pc += 1

    print(time.time() - start, "seconds")
    print(result)
    result = np.array(result)
    print("for init, ", np.sum(result == 1), "-- win, ",
          np.sum(result == -1), "-- lose, ", np.sum(result == 0), "-- draw")
