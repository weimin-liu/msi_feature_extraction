import cv2
import pickle
import tkinter as tk
from tkinter import filedialog

from src.mfe.util import CorSolver


class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.points.append((x, y))


def select_coordinates():
    coordinates = []

    for i in range(3):
        x = int(input(f"Enter the x-coordinate of point {i+1} (Hint: Integer value): "))
        y = int(input(f"Enter the y-coordinate of point {i+1} (Hint: Integer value): "))
        coordinates.append([x, y])

    return coordinates


def select_image():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    return cv2.imread(file_path)


def save_cor_solver(cor_solver):
    root = tk.Tk()
    root.withdraw()

    save_path = filedialog.asksaveasfilename(defaultextension=".pkl")

    with open(save_path, 'wb') as f:
        pickle.dump(cor_solver, f)


def main():
    # Select destination points
    print("Please provide the coordinates for three points on the image.")
    dst_points = select_coordinates()

    coordinate_store = CoordinateStore()

    # Select and process the image
    print("Please select an image file.")
    image = select_image()
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", coordinate_store.select_point)

    print("Please click on the image to select three corresponding points.")
    print("Press 'c' key to continue.")

    while True:
        cv2.imshow("image", gray_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            break

    cv2.destroyAllWindows()

    src_points = coordinate_store.points

    # Fit the CorSolver
    cor_solver = CorSolver()
    cor_solver.fit(dst_points, src_points)

    # Save the CorSolver
    print("Please choose a location to save the CorSolver object.")
    save_cor_solver(cor_solver)


if __name__ == '__main__':
    main()
