import cv2 as cv
import numpy as np

from src.helpers import show_image

def auto_canny(image, sigma=0.5):
    v = np.median(image)
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return cv.Canny(image, lower, upper)


def board_preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.medianBlur(gray, 11)
    threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blockSize=15, C=2)

    # apply morphological operations to close small gaps and
    # unify the outer structure
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    closed = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)

    return closed


def tile_preprocess(tile: np.ndarray) -> np.ndarray:
    tile = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    tile = cv.medianBlur(tile, 11)
    _, tile = cv.threshold(tile, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # tile = auto_canny(tile)

    return tile