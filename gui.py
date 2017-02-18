import numpy as np
import numba
import multiprocessing as mp
import time
import main
from evaluation import *
import cv2
N = 7
M = 4
Fsize = N * N

SIZE = 490
state = False

def on_mouse(event, x, y, flag, params):
    global state
    image, winname, points, board = params
    if not state:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        # get nearest neighbor
        p = np.array((x, y))
        tmp = points - p
        idx = np.argmin(np.sum(np.square(tmp), axis=1))
        p = points[idx, :]
        y, x = (p - 35) // 70
        if board[x, y] != 0:
            return
        board[x, y] = -1
        cv2.circle(image, center=tuple(p.tolist()), radius=20, color=0, thickness=-1)
        cv2.imshow(winname, image)
        cv2.waitKey(1)

        state = not state

def play(model):
    board = np.zeros((N, N), dtype=np.int8)
    turn = True

    global state
    image = np.ones((SIZE, SIZE), dtype=np.uint8) * 180
    points = []
    for i in range(N):
        cv2.line(image, (0, 35 + 70 * i), (SIZE-1, 35 + 70 * i), color=0, thickness=3)
        cv2.line(image, (35 + 70 * i, 0), (35 + 70 * i, SIZE-1), color=0, thickness=3)

        for j in range(N):
            points.append((35 + 70 * i, 35 + 70 * j))

    points = np.array(points)

    cv2.namedWindow('window', flags=cv2.WINDOW_AUTOSIZE)
    if state:
        cv2.imshow('window', image)
        cv2.setMouseCallback('window', on_mouse, [image, 'window', points, board])

    reward = 0
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            return

        if not state:
            if main.winning(-board) or reward != 0:
                key = cv2.waitKey(1)
                return
            board, reward = com_turn(image, board, points)
            cv2.imshow('window', image)

            if reward == 0:
                state = not state
                cv2.setMouseCallback('window', on_mouse, [image, 'window', points, board])

def com_turn(image, board, points):
    actions = np.array(np.where(board == 0))

    # r = np.argmax(model.get(features)[:, 0])
    r = main.getMove(board, model, True, depth=2)
    action = actions[:, r]

    Reward = main.reward(board, action)

    # put
    board[action[0], action[1]] = 1

    # all masses are filled, win
    p = action * 70 + 35
    p = p.tolist()
    p.reverse()
    cv2.circle(image, center=tuple(p), radius=20, color=255, thickness=-1)

    return (board, Reward)

if __name__ == '__main__':
    model = main.MyChain()
    serializers.load_npz('./params/10000.model', model)

    while True:
        # human: True, computer: False
        state = False

        play(model=model)
        arg = input('continue? yes[y]/no[n]\n >>> ')
        if arg == 'y' or arg == 'ye' or arg == 'yes':
            continue
        else:
            break
