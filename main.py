import numpy as np

# NxN bang
N = 3

def winning(board):
    for i in range(N):
        # white
        if (board[i, :] == 1).all():
            return True
        elif (board[:, i] == 1).all():
            return True
    else:
        if (board.diagonal() == 1).all():
            return True
        elif (np.rot90(board).diagonal() == 1).all():
            return True
        else:
            return False

def game(weights):
    board = np.zeros((N, N), dtype=np.int8)
    turn = True
    end = False

    while (board == 0).any():
        # change black and white
        if not turn:
            board = -board

        # can move
        places = np.where(board == 0)

        # set algorithm here.
        nextBoard = []
        for i in range(places[0].size):
            tmp = board.copy()
            tmp[places[0][i], places[1][i]] = 1
            nextBoard.append(tmp.flatten())

        nextBoard = np.array(nextBoard)

        # 0 0 0 1 -1 1 0 0 0' s array
        # 0 0 0
        # 1-1 1
        # 0 0 0
        r = np.argmax(weights.dot(nextBoard.transpose()))

        # r = np.random.randint(places[0].size)

        # put
        board[places[0][r], places[1][r]] = 1

        # winning state
        if winning(board):
            if turn:
                print("white")
                end = True
            else:
                print("black")
                end = True

        # restore black and white
        if not turn:
            board = -board

        print(board)

        if end:
            exit(0)

        # end of this turn
        turn = not turn

    # all masses are filled.
    else:
        print("draw")

if __name__ == '__main__':
    # board 0: blank, 1: white, -1: black
    weights = np.random.rand(1, N*N)
    game(weights)
