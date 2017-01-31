import numpy as np
import numba
import multiprocessing as mp

# NxN bang
# M moku
N = 10
M = 5

@numba.jit
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

@numba.jit
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
        places = np.where(board == 0)

        # set algorithm here.

        # as feature vector
        Boards = []
        for i in range(places[0].size):
            tmp = board.copy()
            tmp[places[0][i], places[1][i]] = 1
            Boards.append(tmp.flatten())

        Boards = np.array(Boards)

        # 0 0 0 1 -1 1 0 0 0' s array
        # 0 0 0
        # 1-1 1
        # 0 0 0
        r = np.argmax(weights.dot(Boards.transpose()))

        # r = np.random.randint(places[0].size)

        # put
        board[places[0][r], places[1][r]] = 1

        Reward = 0
        # winning state
        win = winning(board.flatten())
        if win:
            if turn:
                # print("white")
                Reward = 1
            else:
                # print("black")
                Reward = -1

        # update weights
        if Reward != 0:
            weights += alpha * (Reward - weights.dot(board.flatten())) * board.reshape(1, board.size)
            return weights

        # all masses are filled.
        elif (board != 0).all():
            # print("draw")
            weights += alpha * (Reward - weights.dot(board.flatten())) * board.reshape(1, board.size)
            return weights

        else:
            nextboard = board.copy()
            nextboard = -nextboard

            # can move
            places = np.where(nextboard == 0)

            # set algorithm here.
            Boards = []
            for i in range(places[0].size):
                tmp = nextboard.copy()
                tmp[places[0][i], places[1][i]] = 1
                Boards.append(tmp.flatten())

            Boards = np.array(Boards)

            weights += alpha * (Reward + gamma * np.max(weights.dot(Boards.transpose())) - weights.dot(board.flatten())) * board.reshape(1, board.size)



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
        places = np.where(board == 0)

        # set algorithm here.
        Boards = []
        for i in range(places[0].size):
            tmp = board.copy()
            tmp[places[0][i], places[1][i]] = 1
            Boards.append(tmp.flatten())

        Boards = np.array(Boards)

        if turn:
            r = np.argmax(weights1.dot(Boards.transpose()))
        else:
            r = np.argmax(weights2.dot(Boards.transpose()))

        # put
        board[places[0][r], places[1][r]] = 1

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

        # print(places[0][r], places[1][r])
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

def main(weights):
    weights0 = weights.copy()

    # reinforced learning
    for i in range(50):
        # print(weights)
        weights = game(weights)

    else:
        # display result
        b, res = play(weights0, weights)
        dispBoard(b)
        return res

if __name__ == '__main__':

    # board 0: blank, 1: white, -1: black
    result = []

    testSize = 4
    pool = mp.Pool(testSize)
    args = np.random.rand(1, N*N, testSize)
    result = pool.map(main, iterable=[args[:, :, i] for i in range(testSize)])

    print(result)
