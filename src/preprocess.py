import cv2 as cv
import numpy as np


def board_edge_preprocess(image: np.ndarray, blur: int = 11) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, blur)
    threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize=15, C=2)

    # apply morphological operations to close small gaps and
    # unify the outer structure
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    closed = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)

    return closed


def board_tile_preprocess(board: np.ndarray) -> np.ndarray:
    board = cv.cvtColor(board, cv.COLOR_BGR2GRAY)
    board = cv.medianBlur(board, 7)

    return board


def tile_identification_preprocess(tile: np.ndarray) -> np.ndarray:
    tile = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    tile = cv.medianBlur(tile, 9)
    _, tile = cv.threshold(tile, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return tile
