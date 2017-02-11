import cv2
import numpy as np

SIZE = 490

def on_mouse(event, x, y, flag, params):
    image, winname, points = params
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        # get nearest neighbor
        p = np.array((x, y))
        tmp = points - p
        idx = np.argmin(np.sum(np.square(tmp), axis=1))
        p = points[idx, :]
        cv2.circle(image, center=tuple(p.tolist()), radius=20, color=255, thickness=-1)
        cv2.imshow(winname, image)


def main():
    image = np.ones((SIZE, SIZE), dtype=np.uint8) * 180
    points = []
    for i in range(7):
        cv2.line(image, (0, 35 + 70 * i), (SIZE-1, 35 + 70 * i), color=0, thickness=3)
        cv2.line(image, (35 + 70 * i, 0), (35 + 70 * i, SIZE-1), color=0, thickness=3)

        for j in range(7):
            points.append((35 + 70 * i, 35 + 70 * j))

    points = np.array(points)
    print(points.shape)

    cv2.namedWindow('window', flags=cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('window', on_mouse, [image, 'window', points])
    cv2.imshow('window', image)
    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            return


if __name__ == '__main__':
    main()
