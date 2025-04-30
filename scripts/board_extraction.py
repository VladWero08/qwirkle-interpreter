import cv2 as cv
import numpy as np

from helpers import show_image
from preprocess import board_preprocess

def get_largest_contour_corners(image: np.ndarray) -> np.ndarray:
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    # search for the largest contour
    # that can fit a 4 cornered polygon
    for contour in contours:
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        corners = cv.approxPolyDP(contour, 0.020 * perimeter, True)

        if area > max_area and len(corners) == 4:
            max_area = area
            max_corners = corners

    # reshape from (4, 1, 2) to (4, 2)
    max_corners = max_corners.reshape(-1 ,2)

    return max_corners


def get_padded_corners(corners: np.ndarray, image: np.ndarray, padding: int = 50) -> np.ndarray:
    # get the maximum number of pixels
    # on both X and Y axis
    img_x_max = image.shape[1]; img_y_max = image.shape[0]

    # try adding padding to each corner s.t
    # it will not exceed the boundaries of the image
    top_left = [max(0, corners[0][0] - padding).item(), max(0, corners[0][1] - padding).item()]
    top_right = [min(img_x_max, corners[1][0] + padding).item(), max(0, corners[1][1] - padding).item()]
    bottom_right = [min(img_x_max, corners[2][0] + padding).item(), min(img_y_max, corners[2][1] + padding).item()]
    bottom_left = [max(0, corners[3][0] - padding).item(), min(img_y_max, corners[3][1] + padding).item()]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def get_ordered_corners(corners: np.ndarray) -> np.ndarray:
    idx = np.zeros(4).astype(np.int8)

    summ = np.sum(corners, axis=1)
    diff = np.diff(corners, axis=1)

    idx[0] = np.argmin(summ)
    idx[2] = np.argmax(summ)
    idx[1] = np.argmin(diff)
    idx[3] = np.argmax(diff)

    return corners[idx]


def get_board(
    image: np.ndarray, 
    width: int = 1600,
    height: int = 1600,
    padding: int = 50,    
) -> np.ndarray:
    # get the preprocessed image, gray scaled, thresholded
    threshold = board_preprocess(image)

    # add the padding to the width and height of the cropp
    width = width + 2 * padding
    height = height + 2 * padding

    # extract the corners of the board
    corners = get_largest_contour_corners(threshold)
    corners = get_ordered_corners(corners)
    corners = get_padded_corners(corners, image, padding)

    centered_board = np.array([
        [0, 0], 
        [width - 1, 0], 
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(corners, centered_board)
    image_cropped = cv.warpPerspective(image, M, (width, height))

    return image_cropped


if __name__ == "__main__":
    image = cv.imread("../playground/board.jpg")
    board = get_board(image)

    show_image(board)

# visualization = original.copy()
# colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
# for i, point in enumerate(max_corners):
#     cv.circle(visualization, tuple(point), 10, colors[i], -1)
#     cv.putText(visualization, f"Point {i}", (point[0]+10, point[1]+10), 
#                cv.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)
# show_image(visualization, "Source Points")