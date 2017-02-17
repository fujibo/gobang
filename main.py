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
Fsize = N * N

@numba.jit
def winning(board):
    b = board.reshape(N, N) == 1
    for i in range(N - M + 1):
        for j in range(N - M + 1):
            tmp = b[i:i + M, j:j + M].flatten()
            for k in range(N - M + 1):
                # col
                if tmp[k] and tmp[k+4] and tmp[k+8] and tmp[k+12]:
                    return True
                # row
                if tmp[k*4] and tmp[k*4+1] and tmp[k*4+2] and tmp[k*4+3]:
                    return True

            # diag
            else:
                if tmp[0] and tmp[5] and tmp[10] and tmp[15]:
                    return True
                elif tmp[3] and tmp[6] and tmp[9] and tmp[12]:
                    return True
    else:
        return False

# @numba.jit(numba.b1(numba.i1[:]))
# def winning(board):
#     'board: flatten board'
#     tf = False
#
#     board = board.reshape(N, N)
#     for i in range(N):
#         if np.sum(board[i, :] == 1) < 4:
#             continue
#         for j in range(N - M + 1):
#             # white
#             if (board[i, j:j + M] == 1).all():
#                 tf = True
#                 break
#     if tf:
#         return True
#     for i in range(N):
#         if np.sum(board[:, i] == 1) < 4:
#             continue
#         for j in range(N - M + 1):
#             if (board[j:j + M, i] == 1).all():
#                 tf = True
#                 break
#     if tf:
#         return True
#
#     for i in range(N - M + 1):
#         for j in range(N - M + 1):
#             tmp = board[i:i + M, j:j + M]
#             if (tmp.diagonal() == 1).all():
#                 tf = True
#                 # print("diag1 i, j", i, j)
#                 break
#
#             elif (np.rot90(tmp).diagonal() == 1).all():
#                 tf = True
#                 # print("diag2 i, j", i, j)
#                 break
#     return tf


@numba.jit(numba.f4(numba.i1[:, :], numba.i8[:, :]))
def reward(board, action):
    tmp = board.copy()
    tmp[action[0], action[1]] = 1
    reward = np.zeros(1, dtype=np.float32)
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
        actions.shape[1], axis=0).astype(np.float32)
    for i in range(actions.shape[1]):
        Features[i, actions[0, i] + actions[1, i] * N] = 1
    # Features = np.zeros((actions.shape[1], Fsize), dtype=np.float32)
    # for i in range(actions.shape[1]):
    #     Features[i, :] = getFeature(board, actions[:, i])
    return Features

# @numba.jit(numba.f8[:](numba.f8[:]))


def game(model):

    xs = []
    ys = []
    # parameters
    gamma = 0.9
    epsilon = 0.10

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
            ys.append(Reward)

            return (xs, ys)

        # else
        else:
            nextboard = board.copy()
            nextboard[action[0], action[1]] = 1
            # can move
            nextboard = -nextboard
            nextactions = np.array(np.where(nextboard == 0))
            # set algorithm here.
            nextfeatures = getFeatures(nextboard, nextactions)

            xs.append(feature)

            # この内の最大となるyを選択したい
            y = -gamma * np.max(model.get(nextfeatures)[:, 0])
            ys.append(y)

        # put
        board[action[0], action[1]] = 1

        # restore black and white
        if not turn:
            board = -board

        # print(board)

        # end of this turn
        turn = not turn


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
        features = getFeatures(board, actions)

        if turn:
            r = np.argmax(weights1.dot(features.transpose()))
        else:
            r = np.argmax(weights2.dot(features.transpose()))

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
        # print(board)

        # end of the game
        if Reward != 0:
            return (board, Reward, moved)

        # all masses are filled.
        elif (board != 0).all():
            print("draw")
            return (board, Reward, moved)

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
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    losses = []
    plt.hold(False)

    x_data = []
    y_data = []
    data_size = 0

    for i in range(1, 10001):

        xs, ys = game(model)
        num = len(ys)

        x_data += xs
        y_data += ys
        data_size += num

        if i % 1 == 0:
            model.cleargrads()
            # a x 49
            x_ = Variable(np.array(x_data, dtype=np.float32).reshape(data_size, Fsize))
            # a x 1
            y_ = Variable(np.array(y_data, dtype=np.float32).reshape(data_size, 1))
            loss = model(x_, y_)
            loss.backward()
            optimizer.update()

            losses.append(loss.data)

        if i % 5 == 0:
            test(model)

            plt.plot(losses, 'b')
            plt.yscale('log')
            plt.pause(0.01)

        if i % 1000 == 0:
            serializers.save_npz('./params/{}.model'.format(i), model)

    # # reinforced learning
    # for i in range(1000):
    #     if i % 10 == 0:
    #         weights0 = weights.copy()
    #         print(weights0)
    #         test(weights0)
    #     if i % 20 == 5:
    #         pstart = time.time()
    #         b, res, moved = play(weights0, weights)
    #         dispBoard(b)
    #         print(moved)
    #         print("play time", time.time() - pstart)
    #
    #     if i % 100 == 0:
    #         # np.save('weights{}_{}.npy'.format(i, pid), weights)
    #         print(weights)
    #         print(i)
    #     if np.max(np.abs(weights)) > 1000:
    #         queue.put(-100)
    #         return
    #     weights = game(weights)
    # else:
    #     # display result
    #     np.save('weights1000.npy', weights)
    #     pstart = time.time()
    #     b, res, moved = play(weights0, weights)
    #     dispBoard(b)
    #     print(moved)
    #     print("play time", time.time() - pstart)
    #     queue.put(res)
    #     # return res
    queue.put(1)
    return

@numba.jit(numba.i8(numba.i1[:, :], numba.f8[:], numba.b1, numba.i8))
def getMove(board, model, flag, depth):
    '''board, model, flag, depth
    flag: if this function is for idx or for score'''
    # check mate
    if flag:
        res = Mate(board, True, depth=3)
        if res != -1:
            return res

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
            board[actions[0, i], actions[1, i]] = 1
            score[i] = getMove(-board, model, False, depth-1)
            board[actions[0, i], actions[1, i]] = 0
        else:
            if flag:
                return np.argmax(score)
            else:
                return -np.max(score)

@numba.jit(numba.i8(numba.i1[:, :], numba.b1, numba.i8))
def Mate(board, flag, depth):
    '''board, model, flag, depth
    flag: if this function is for idx or for score'''
    if depth == 1:
        actions = np.array(np.where(board == 0))
        for i in range(actions.shape[1]):
            board[actions[0, i], actions[1, i]] = 1
            if winning(board.flatten()):
                board[actions[0, i], actions[1, i]] = 0
                if flag:
                    return i
                else:
                    return -1
            board[actions[0, i], actions[1, i]] = 0
        else:
            if flag:
                return -1
            else:
                return 0

    else:
        # 相手の手によって決着がつかないことを知っていれば舐めた手をしても良い
        if flag:
            res = Mate(board, True, depth=1)
            if res != -1:
                return res

        actions = np.array(np.where(board == 0))
        score = np.zeros(actions.shape[1])
        for i in range(actions.shape[1]):
            board[actions[0, i], actions[1, i]] = 1
            score[i] = Mate(-board, False, depth-1)
            board[actions[0, i], actions[1, i]] = 0
            if score[i] == 1:
                if flag:
                    return i
                else:
                    return -1
        else:
            if flag:
                return -1
            else:
                return -np.max(score)

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
