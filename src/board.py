import cv2 as cv
import numpy as np
import os

from src.preprocess import board_edge_preprocess, board_tile_preprocess
from src.tiles import get_similarity_sift, get_tile_type, TWO_TOP_LEFT
from src.helpers import show_image, read_template

def get_tile_template(path: list[str]) -> np.ndarray:
    tile = read_template(path)
    tile = board_tile_preprocess(tile)

    return tile

QUADRANT_TILE = 3
QUADRANT_1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 1, QUADRANT_TILE, 0],
    [0, 0, 0, 0, 1, QUADRANT_TILE, 1, 0],
    [0, 0, 0, 1, QUADRANT_TILE, 1, 0, 0],
    [0, 0, 1, QUADRANT_TILE, 1, 0, 0, 0],
    [0, 1, QUADRANT_TILE, 1, 0, 0, 0, 0],
    [0, QUADRANT_TILE, 1, 0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])
QUADRANT_2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, QUADRANT_TILE, 1, 0, 0, 0, 2, 0],
    [0, 1, QUADRANT_TILE, 1, 0, 0, 0, 0],
    [0, 0, 1, QUADRANT_TILE, 1, 0, 0, 0],
    [0, 0, 0, 1, QUADRANT_TILE, 1, 0, 0],
    [0, 0, 0, 0, 1, QUADRANT_TILE, 1, 0],
    [0, 2, 0, 0, 0, 1, QUADRANT_TILE, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
])

DIR = os.path.dirname(os.path.abspath(__file__))
CIRCLE = get_tile_template([DIR, "..", "templates", "circle.jpg"])
DIAMOND = get_tile_template([DIR, "..", "templates", "diamond.jpg"])
FLOWER = get_tile_template([DIR, "..", "templates", "flower.jpg"])
SQUARE = get_tile_template([DIR, "..", "templates", "square.jpg"])
STAR = get_tile_template([DIR, "..", "templates", "star.jpg"])
SUN = get_tile_template([DIR, "..", "templates", "sun.jpg"])
TILE_TEMPLATES = [(CIRCLE, "circle"), (DIAMOND, "diamond"), (FLOWER, "flower"), (STAR, "star"), (SUN, "sun"), (SQUARE, "square")]

NO_OF_TILES_PER_ROW = 16
NO_OF_TILES_PER_COL = 16


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
    top_left = [max(0, corners[0][0] - padding), max(0, corners[0][1] - padding)]
    top_right = [min(img_x_max, corners[1][0] + padding), max(0, corners[1][1] - padding)]
    bottom_right = [min(img_x_max, corners[2][0] + padding), min(img_y_max, corners[2][1] + padding)]
    bottom_left = [max(0, corners[3][0] - padding), min(img_y_max, corners[3][1] + padding)]

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
    padding: int = 0,    
) -> np.ndarray:
    # get the preprocessed image, gray scaled, thresholded
    threshold = board_edge_preprocess(image)

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


def get_scoring_board(patches: np.ndarray) -> int:
    scoring_board = []
    quadrants = [(1, 1), (1, 9), (9, 1), (9, 9)]

    for row, col in quadrants:
        two_top_left = patches[16 * row + col]

        if get_similarity_sift(two_top_left, TWO_TOP_LEFT) > 0.5:
            scoring_board.append(QUADRANT_1)
        else:
            scoring_board.append(QUADRANT_2)

    scoring_board = np.array(scoring_board)
    scoring_board = np.array([np.hstack(scoring_board[:2]), np.hstack(scoring_board[2:])])
    scoring_board = np.vstack(scoring_board)

    return scoring_board


def get_board_tile_type_initial(scoring_board: np.ndarray, patches: np.ndarray) -> list:
    tile_board = []

    for i in range(NO_OF_TILES_PER_ROW):
        row = []

        for j in range(NO_OF_TILES_PER_COL):
            if scoring_board[i][j] == QUADRANT_TILE:
                tile_type = get_tile_type(patches[NO_OF_TILES_PER_ROW * i + j])
                row.append(tile_type)
            else:
                row.append(None)

        tile_board.append(row)

    return tile_board


def get_board_tiles_positions(
    board: np.ndarray,
    padding: int = 0,
    threshold: float = 0.7,
) -> set:
    board = board_tile_preprocess(board)
    # board_copy = board.copy()
    board_tiles = [[0 for i in range(NO_OF_TILES_PER_COL)] for j in range(NO_OF_TILES_PER_ROW)]
    matched_tiles = set()

    for tile_template, _ in TILE_TEMPLATES:
        res = cv.matchTemplate(board, tile_template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        w, h = tile_template.shape[::-1]

        for pt in zip(*loc[::-1]):
            center = (pt[1] + h // 2 - padding, pt[0] + w // 2 - padding)
            patch = (center[0] // 100, center[1] // 100)

            # cv.rectangle(board_copy, pt, (pt[0] + w, pt[1] + h), 255, 2)
            matched_tiles.add(patch)

    # show_image(board_copy)

    try:
        for (i, j) in matched_tiles:
            board_tiles[i][j] = 1
    except IndexError as e:
        print(i, j, "failed")


    return board_tiles


def get_board_simple_shapes(
    board: np.ndarray,
    shapes: list[tuple] = [(SQUARE, "square"), (CIRCLE, "circle"), (DIAMOND, "diamond")],
    threshold: float = 0.85,
    padding: int = 0,    
) -> dict:
    board = board_tile_preprocess(board)
    positions = dict()

    for tile_template, tile_template_name in shapes:
        res = cv.matchTemplate(board, tile_template, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        w, h = tile_template.shape[::-1]

        for pt in zip(*loc[::-1]):
            center = (pt[1] + h // 2 - padding, pt[0] + w // 2 - padding)
            patch = (center[0] // 100, center[1] // 100)

            if patch not in positions:
                positions[patch] = tile_template_name

    return positions