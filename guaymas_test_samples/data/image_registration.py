"""
https://stackoverflow.com/questions/58494837/get-the-numpy-index-for-a-given-image-location
"""
import re

import cv2
import numpy as np
import pickle
import pandas as pd
import re
import matplotlib.pyplot as plt

from src.mfe.util import CorSolver


class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.points.append((x, y))


if __name__ == '__main__':

    dst_points = [[30, 128], [31, 26], [478,22]]

    coordinateStore1 = CoordinateStore()
    # Read an image
    image = cv2.imread('2022_06_07_5000_0-4cm_100um_A_072.tif')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Set up window and mouse callback function
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", coordinateStore1.select_point)

    # Loop until the 'c' key is pressed
    while True:

        # Display image; wait for keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # If 'c' key is pressed, break from loop
        if key == ord("c"):
            break

    cv2.destroyAllWindows()

    src_points = coordinateStore1.points
    corSolver = CorSolver()
    corSolver.fit(dst_points, src_points)

    with open('./corSolver.pkl', 'wb') as f:
        pickle.dump(corSolver, f)
    image = image.astype(int)
    spot_list = pd.read_csv('./roi_spot.txt', sep=' ')
    spot_list['x'] = spot_list['spot-name'].apply(lambda x: int(re.findall(r'R00X(\d+)', x)[0]))
    spot_list['y'] = spot_list['spot-name'].apply(lambda x: int(re.findall(r'Y(\d+)', x)[0]))
    spot_list[['xa', 'ya']] = corSolver.transform(spot_list[['x', 'y']].to_numpy())
    spot_list[['xa', 'ya']] = spot_list[['xa', 'ya']].astype(int)
    spot_list['c'] = spot_list.apply(lambda x: image[x['ya'], x['xa']], axis=1)

    left = spot_list['xa'].min()
    right = spot_list['xa'].max()
    up = spot_list['ya'].min()
    down = spot_list['ya'].max()

    spot_list = spot_list[['spot-name', 'x', 'y', 'c']]
    with open('./spot_list.csv', 'w') as f:
        spot_list.to_csv(f, index=False)

    image_region = image[up:down, left:right]
    plt.imshow(image_region, cmap='gray')
    plt.show()
