import numpy as np
import numba
import multiprocessing as mp
import time
# NxN bang
# M moku
# board 0: blank, 1: white, -1: black
N = 10
M = 5
Fsize = N*N * (N*N-1) *3 // 2 + N*N

@numba.jit(numba.b1(numba.i1[:]))
def winning(board):
    tf = False

    board = board.reshape(N, N)
    for i in range(N):
        for j in range(N-M+1):
            # white
            if (board[i, j:j+M] == 1).all():
                tf = True
                # print("row i, j", i, j)
                break
            elif (board[j:j+M, i] == 1).all():
                tf = True
                # print("col i, j", i, j)
                break
    if tf:
        return True

    for i in range(N-M+1):
        for j in range(N-M+1):
            tmp = board[i:i+M, j:j+M]
            if (tmp.diagonal() == 1).all():
                tf = True
                # print("diag1 i, j", i, j)
                break

            elif (np.rot90(tmp).diagonal() == 1).all():
                tf = True
                # print("diag2 i, j", i, j)
                break
    return tf

@numba.jit(numba.f8(numba.i1[:, :], numba.i8[:, :]))
def reward(board, action):
    tmp = board.copy()
    tmp[action[0], action[1]] = 1
    reward = 0
    # winning state
    if winning(tmp.flatten()):
        reward = 10.0
    elif (tmp != 0).all():
        reward = -1.0
    return reward

@numba.jit(numba.i1[:](numba.i1[:, :], numba.i8[:, :]))
def getFeature(board, action):
    'board: board now, action: one action'
    # 0 0 0 1 -1 1 0 0 0' s array
    # 0 0 0
    # 1-1 1
    # 0 0 0

    tmp = board.copy()
    tmp[action[0], action[1]] = 1

    tmp = tmp.flatten()

    # use two masses relation as parameters
    # (N*N-1)*(N*N)/2  x 3 <all the combinations> x <(WW, BB), (WB, BW), blank>
    relation = np.zeros(((N*N) * (N*N-1)//2, 3), dtype=np.int8)
    i = 0
    for mass_idx in range(tmp.size):
        # white
        size = tmp[mass_idx+1:].size
        if tmp[mass_idx] == 1:
            relation[i:i+size, 0] = (tmp[mass_idx+1:] == 1)
            relation[i:i+size, 1] = (tmp[mass_idx+1:] == -1)
            relation[i:i+size, 2] = (tmp[mass_idx+1:] == 0)
            i += size
        # black
        elif tmp[mass_idx] == -1:
            relation[i:i+size, 1] = -(tmp[mass_idx+1:] == 1).astype(np.int8)
            relation[i:i+size, 0] = -(tmp[mass_idx+1:] == -1).astype(np.int8)
            relation[i:i+size, 2] = -(tmp[mass_idx+1:] == 0).astype(np.int8)
            i += size
        # blank
        else:
            relation[i:i+size, 2] = tmp[mass_idx+1:]
            i += size

    # use future as a parameter
    feature = np.hstack((tmp, relation.flatten()))
    return feature

@numba.jit(numba.i1[:, :](numba.i1[:, :], numba.i8[:, :]))
def getFeatures(board, actions):
    'board: board now, actions: can put there'
    # use next board(after-an-action) state  as parameters
    Features = np.zeros((actions.shape[1], Fsize), dtype=np.int8)
    for i in range(actions.shape[1]):
        Features[i, :] = getFeature(board, actions[:, i])
    return Features

# @numba.jit(numba.f8[:](numba.f8[:]))
def game(weights):

    # parameters
    alpha = 0.003
    gamma = 0.9

    board = np.zeros((N, N), dtype=np.int8)
    turn = True

    while True:
        # change black and white
        if not turn:
            board = -board

        # can move
        actions = np.array(np.where(board == 0))

        # set algorithm here.

        # as feature vectors
        features = getFeatures(board, actions)
        r = np.argmax(weights.dot(features.transpose()))
        # r = np.random.randint(actions[0].size)

        action = actions[:, r]
        feature = features[r, :]

        Reward = reward(board, action)

        # update weights

        # all masses are filled, win
        if Reward != 0:
            diff = Reward - weights.dot(feature)
            weights += alpha * diff * feature.reshape(1, feature.size)
            return weights


        # else
        else:
            nextboard = board.copy()
            nextboard[action[0], action[1]] = 1
            # can move
            nextboard = -nextboard
            nextactions = np.array(np.where(nextboard == 0))
            # set algorithm here.
            nextfeatures = getFeatures(nextboard, nextactions)

            diff = (Reward - gamma * np.max(weights.dot(nextfeatures.transpose())) - weights.dot(feature))
            # reward - (hoge) because of opponent turn
            weights += alpha * diff * feature.reshape(1, feature.size)

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
    for i in range(N):
        for j in range(N):
            if board[i, j] == 1:
                print("O", end="")
            elif board[i, j] == -1:
                print("X", end="")
            else:
                print(" ", end="")
        else:
            print("")

def main(queue, weights, pid):
    weights0 = weights.copy()

    # reinforced learning
    for i in range(1000):
        if i % 10 == 0:
            weights0 = weights.copy()
            print(weights0)
            test(weights0)
        if i % 20 == 5:
            pstart = time.time()
            b, res, moved = play(weights0, weights)
            dispBoard(b)
            print(moved)
            print("play time", time.time() - pstart)

        if i % 100 == 0:
            # np.save('weights{}_{}.npy'.format(i, pid), weights)
            print(weights)
            print(i)
        if np.max(np.abs(weights)) > 1000:
            queue.put(-100)
            return
        weights = game(weights)
    else:
        # display result
        # np.save('weights10000_{}.npy'.format(pid), weights)
        pstart = time.time()
        b, res, moved = play(weights0, weights)
        dispBoard(b)
        print(moved)
        print("play time", time.time() - pstart)
        queue.put(res)
        # return res

def test(weights):
    b = np.zeros((N, N), dtype=np.int8)
    b[2, 2] = 1
    b[2, 3] = 1
    b[2, 4] = 1
    b[2, 5] = 1
    a = np.array(np.where(b == 0))
    fs = getFeatures(b, a)

    score = weights.dot(fs.transpose())
    print(score)
    print(np.argmax(score), "<- idx, ", np.max(score), "<- value")
    print(a[:, np.argmax(score)])

if __name__ == '__main__':

    result = []

    testSize = 1
    queue = mp.Queue()
    # ps = [mp.Process(target=main, args=(queue, np.random.rand(1, Fsize)/10, i)) for i in range(testSize)]
    ps = [mp.Process(target=main, args=(queue, np.zeros((1, Fsize)), i)) for i in range(testSize)]

    start = time.time()
    pc = 0 # work as program counter
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
    print("for init, ", np.sum(result==1), "-- win, ", np.sum(result==-1), "-- lose, ", np.sum(result==0), "-- draw")
