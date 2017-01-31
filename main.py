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

if __name__ == '__main__':
    # board 0: blank, 1: white, -1: black
    board = np.zeros((N, N), dtype=np.int8)
    turn = True
    end = False
    while (board == 0).any():
        # change black and white
        if not turn:
            board = -board
        # put
        places = np.where(board == 0)
        r = np.random.randint(places[0].size)
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
