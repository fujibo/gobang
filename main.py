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

    # parameters
    alpha = 0.5
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
        if winning(board):
            if turn:
                print("white")
                Reward = 1
            else:
                print("black")
                Reward = -1

        # update weights
        if Reward != 0:
            weights += alpha * (Reward - weights.dot(board.flatten())) * board.reshape(1, board.size)
            return weights

        # all masses are filled.
        elif (board == 0).any():
            print("draw")
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


if __name__ == '__main__':
    weights = np.random.rand(1, N*N)
    # board 0: blank, 1: white, -1: black
    for i in range(10000):
        print(weights)
        weights = game(weights)
