import os
import cv2 as cv
import numpy as np

from src.helpers import show_image, read_template
from src.preprocess import tile_preprocess

def get_tile_template(path: list[str]) -> np.ndarray:
    tile = read_template(path)
    tile = tile_preprocess(tile)

    return tile

DIR = os.path.dirname(os.path.abspath(__file__))
TWO_TOP_LEFT = read_template([DIR, "..", "templates", "two_top_left.jpg"])
TWO_TOP_RIGHT = read_template([DIR, "..", "templates", "two_top_right.jpg"])

CIRCLE = get_tile_template([DIR, "..", "templates", "circle.jpg"])
DIAMOND = get_tile_template([DIR, "..", "templates", "diamond.jpg"])
FLOWER = get_tile_template([DIR, "..", "templates", "flower.jpg"])
SQUARE = get_tile_template([DIR, "..", "templates", "square.jpg"])
STAR = get_tile_template([DIR, "..", "templates", "star.jpg"])
SUN = get_tile_template([DIR, "..", "templates", "sun.jpg"])
TILE_TEMPLATES = [(CIRCLE, "circle"), (DIAMOND, "diamond"), (FLOWER, "flower"), (STAR, "star"), (SUN, "sun"), (SQUARE, "square")]

NO_OF_TILES_PER_ROW = 16
NO_OF_TILES_PER_COL = 16


def get_patches(
    image: np.ndarray, 
    width: int = 1600,
    height: int = 1600,
    padding: int = 0, 
) -> np.ndarray:
    # compute the patch width and height
    p_height = height // NO_OF_TILES_PER_ROW
    p_width = width // NO_OF_TILES_PER_COL
    target_shape = (p_height + 2 * padding, p_width + 2 * padding, image.shape[2])
    patches = []

    for i in range(NO_OF_TILES_PER_ROW):
        for j in range(NO_OF_TILES_PER_COL):
            # compute the corners of the patch
            row_start = max(0, i * p_height - padding)
            row_end = min(image.shape[0], (i + 1) * p_height + padding)
            col_start = max(0, j * p_width - padding)
            col_end = min(image.shape[1], (j + 1) * p_width + padding) 

            patch = image[row_start:row_end, col_start:col_end]
            padded_patch = np.zeros(target_shape, dtype=patch.dtype)

            h, w = patch.shape[:2]
            padded_patch[:h, :w] = patch
            patches.append(padded_patch)

    return np.array(patches)


def get_similarity_sift(
    image_1: np.ndarray,
    image_2: np.ndarray
) -> float:
    sift = cv.SIFT.create()
    bf = cv.BFMatcher()

    kp_1, des_1 = sift.detectAndCompute(image_1, None)
    kp_2, des_2 = sift.detectAndCompute(image_2, None)

    if des_1 is None or des_2 is None:
        return 0.0

    matches = bf.knnMatch(des_1, des_2, k=2)
    good_matches = 0

    for _match in matches:
        if len(_match) != 2:
            continue
        
        m, n = _match

        if m.distance < 0.75 * n.distance:
            good_matches += 1

    similarity = good_matches / max(len(kp_1), len(kp_2))

    return similarity


def get_similarity_template_matching(tile: np.ndarray, template: np.ndarray) -> float:
    result = cv.matchTemplate(tile, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv.minMaxLoc(result)

    return max_val


def get_tile_shape(tile: np.ndarray) -> str:
    best_match = 0
    best_match_name = None

    # try to get the most similar shape using
    # the SIFT features for each tile
    for tile_template, tile_template_name in TILE_TEMPLATES:
        similarity = get_similarity_sift(tile, tile_template)

        if similarity > best_match:
            best_match = similarity
            best_match_name = tile_template_name

    # if a template was matched, return it
    if best_match_name is not None:
        return best_match_name
    
    # get the most similar shape based
    # on template matching using normed correlation
    for tile_template, tile_template_name in TILE_TEMPLATES:
        similarity = get_similarity_template_matching(tile, tile_template)

        if similarity > best_match:
            best_match = similarity
            best_match_name = tile_template_name


    return best_match_name


def get_dominant_hsv(image: np.ndarray) -> tuple[float]:
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (0, 0, 120), (180, 255, 255))

    hue = hsv[:, :, 0][mask > 0]
    saturation = hsv[:, :, 1][mask > 0]
    value = hsv[:, :, 2][mask > 0]
    
    hue = np.median(hue)
    saturation = np.median(saturation)
    value = np.median(value)

    return hue, saturation, value


def get_tile_color(tile: np.ndarray) -> str:
    hue, saturation, value = get_dominant_hsv(tile)

    if saturation < 30 and value > 200:
        return "white"
    
    if hue < 10 or hue > 160:
        return "red"
    elif 11 <= hue <= 25:
        return "orange"
    elif 26 <= hue <= 35:
        return "yellow"
    elif 36 <= hue <= 85:
        return "green"
    elif 86 <= hue <= 130:
        return "blue"
    else:
        return "unknown"