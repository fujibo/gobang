import numpy as np
import numba
import multiprocessing as mp
import time
# NxN bang
# M moku
# board 0: blank, 1: white, -1: black
N = 7
M = 4
Fsize = N*N * (N*N-1) * 2 + 1 + N*N

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

@numba.jit(numba.f8(numba.i1[:], numba.i8[:]))
def reward(board, action):
    tmp = board.copy()
    tmp[action[0], action[1]] = 1
    reward = 0
    # winning state
    if winning(tmp.flatten()):
        reward = 1
    elif (tmp != 0).all():
        reward = -0.1
    return reward

@numba.jit(numba.i1[:](numba.i1[:], numba.i8[:], numba.b1))
def getFeature(board, action, future):
    'board: board now, action: one action'
    # 0 0 0 1 -1 1 0 0 0' s array
    # 0 0 0
    # 1-1 1
    # 0 0 0

    tmp = board.copy()
    if not future:
        tmp[action[0], action[1]] = 1
    else:
        tmp[action[0], action[1]] = -1

    tmp = tmp.flatten()

    # use two masses relation as parameters
    # (N*N-1)*(N*N)/2  x 4 <all the combinations> x <W-W, W-B, B-B, blank>
    relation = np.zeros(((N*N) * (N*N-1)//2, 4), dtype=np.int8)
    i = 0
    for mass_idx in range(tmp.size):
        # white
        size = tmp[mass_idx+1:].size
        if tmp[mass_idx] == 1:
            relation[i:i+size, 0] = (tmp[mass_idx+1:] == 1)
            relation[i:i+size, 1] = (tmp[mass_idx+1:] == -1)
            relation[i:i+size, 3] = (tmp[mass_idx+1:] == 0)
            i += size
        # black
        elif tmp[mass_idx] == -1:
            relation[i:i+size, 1] = (tmp[mass_idx+1:] == 1)
            relation[i:i+size, 2] = (tmp[mass_idx+1:] == -1)
            relation[i:i+size, 3] = (tmp[mass_idx+1:] == 0)
            i += size
        # blank
        else:
            relation[i:i+size, 3] = tmp[mass_idx+1:]
            i += size

    # use future as a parameter
    feature = np.hstack((tmp, np.array([future]), relation.flatten()))
    return feature

@numba.jit(numba.i1[:](numba.i1[:], numba.i8[:], numba.b1))
def getFeatures(board, actions, future):
    'board: board now, actions: can put there'
    # use next board(after-an-action) state  as parameters
    Features = np.zeros((actions.shape[1], Fsize), dtype=np.int8)
    for i in range(actions.shape[1]):
        Features[i, :] = getFeature(board, actions[:, i], future)
    return Features

@numba.jit(numba.f8[:](numba.f8[:]))
def game(weights):

    # parameters
    alpha = 0.01
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
        features = getFeatures(board, actions, False)
        r = np.argmax(weights.dot(features.transpose()))
        # r = np.random.randint(actions[0].size)

        action = actions[:, r]
        feature = features[r, :]

        Reward = reward(board, action)

        # update weights

        # all masses are filled, win
        if Reward != 0:
            weights += alpha * (Reward - weights.dot(feature)) * feature.reshape(1, feature.size)
            return weights


        # else
        else:
            nextboard = board.copy()
            nextboard[action[0], action[1]] = 1
            # can move
            # nextboard = -nextboard
            nextactions = np.array(np.where(nextboard == 0))
            # set algorithm here.
            nextfeatures = getFeatures(nextboard, nextactions, True)

            weights += alpha * (Reward + gamma * np.min(weights.dot(nextfeatures.transpose())) - weights.dot(feature)) * feature.reshape(1, feature.size)

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
    turn = True

    while True:
        # change black and white
        if not turn:
            board = -board

        # can move
        actions = np.array(np.where(board == 0))
        features = getFeatures(board, actions, False)

        if turn:
            r = np.argmax(weights1.dot(features.transpose()))
        else:
            r = np.argmax(weights2.dot(features.transpose()))

        # put
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
            return (board, Reward)

        # all masses are filled.
        elif (board != 0).all():
            print("draw")
            return (board, Reward)

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

def main(queue, weights):
    weights0 = weights.copy()

    # reinforced learning
    for i in range(100):
        # print(weights)
        if i == 10:
            weights0 = weights.copy()
        weights = game(weights)

    else:
        # display result
        pstart = time.time()
        b, res = play(weights0, weights)
        dispBoard(b)
        print("play time", time.time() - pstart)
        queue.put(res)
        # return res

if __name__ == '__main__':

    result = []

    testSize = 4
    queue = mp.Queue()
    ps = [mp.Process(target=main, args=(queue, np.random.rand(1, Fsize)/10)) for i in range(testSize)]

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
