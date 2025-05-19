import os
import cv2 as cv
import numpy as np

from src.helpers import show_image, read_template
from src.preprocess import tile_identification_preprocess

def get_tile_template(path: list[str]) -> np.ndarray:
    tile = read_template(path)
    tile = tile_identification_preprocess(tile)

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

COLOR_RANGES = {
    'red':    [(np.array([0, 70, 50]), np.array([5, 255, 255])),
               (np.array([160, 70, 50]), np.array([180, 255, 255]))],
    'orange': [(np.array([5, 100, 100]), np.array([25, 255, 255]))],
    'yellow': [(np.array([25, 100, 100]), np.array([35, 255, 255]))],
    'green':  [(np.array([36, 70, 70]), np.array([85, 255, 255]))],
    'blue':   [(np.array([90, 70, 70]), np.array([130, 255, 255]))],
    'white':  [(np.array([0, 0, 200]), np.array([180, 30, 255]))],
}

def get_patches(
    image: np.ndarray, 
    width: int = 1600,
    height: int = 1600,
    padding: int = 0, 
) -> np.ndarray:
    """
    Splits an image into a grid of patches corresponding to tiles, 
    with optional padding around each patch.

    Returns:
    --------
    patches: np.ndarray
        A NumPy array of image patches extracted from the original image.
    """
    # compute the patch width and height
    p_height = height // NO_OF_TILES_PER_ROW
    p_width = width // NO_OF_TILES_PER_COL
    # compute the desired shape of the tile by including the padding
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
    """
    Computes the similarity between two images using SIFT features and
    Lowe's ratio test for good feature matches.

    Returns:
    --------
    similarity: float
        A similarity score between the two images, ranging from 0.0 to 1.0.
    """
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

        # Lowe's ratio test
        if m.distance < 0.75 * n.distance:
            good_matches += 1

    similarity = good_matches / max(len(kp_1), len(kp_2))

    return similarity


def get_similarity_template_matching(tile: np.ndarray, template: np.ndarray) -> float:
    """
    Compares a tile image to a template using normalized cross-correlation 
    template matching.

    Returns:
    --------
    max_val: float
        The maximum correlation value indicating similarity.
    """
    result = cv.matchTemplate(tile, template, cv.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv.minMaxLoc(result)

    return max_val


def get_tile_shape(tile: np.ndarray) -> str:
    """
    Identifies the shape of a tile by comparing it to known tile templates
    using both SIFT feature matching and template matching.

    Returns:
    --------
    best_match_name: str
        The name of the shape that best matches the input tile.
    """
    tile = tile_identification_preprocess(tile)

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


def get_tile_color(tile: np.ndarray) -> str:
    """
    Determines the dominant color of a tile by converting it to HSV
    and analyzing predefined color ranges.

    Returns:
    --------
    dominant_color: str
        The name of the dominant color found in the tile.
    """
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    # initialize total mask and pixel count dictionary
    total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    pixel_counts = {}

    # generate individual masks and count pixels
    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv.inRange(hsv, lower, upper)
        pixel_counts[color] = cv.countNonZero(mask)
        total_mask |= mask
    
    dominant_color = max(pixel_counts, key=pixel_counts.get)

    return dominant_color


def get_tile_type(tile: np.ndarray) -> tuple[str]:
    """
    Determines the type of a tile by identifying both its shape and color.

    Returns:
    --------
    (shape, color): tuple[str]
        A tuple containing the shape and color of the tile.
    """
    shape = get_tile_shape(tile)
    color = get_tile_color(tile)

    return (shape, color)


def check_different_tiles(
    tile_from: np.ndarray,
    tile_to: np.ndarray,
    diff_threshold: int = 30,    
) -> bool:
    """
    Checks if two tiles are different by comparing their grayscale pixel values 
    and computing the amount of pixel change beyond a set threshold.

    Returns:
    --------
    is_different: bool
        True if the tiles are considered different, otherwise False.
    """

    if get_tile_color(tile_to) == "white":
        percentage = 0.6
    else:
        percentage = 0.75

    # compute the number of pixels that need to be
    # changed in order to consider that the tiles are different
    # to be a % of the total number of pixels of a tile
    pixel_threshold = int(tile_from.shape[0] * tile_from.shape[1] * percentage)

    tile_from = cv.cvtColor(tile_from, cv.COLOR_BGR2GRAY)
    tile_to = cv.cvtColor(tile_to, cv.COLOR_BGR2GRAY)

    # compute the absolute difference between the two tiles
    diff = cv.absdiff(tile_from, tile_to)
    # threshold the difference
    _, diff_mask = cv.threshold(diff, diff_threshold, 255, cv.THRESH_BINARY)

    # count how many pixels have changed
    pixel_changes = int(np.count_nonzero(diff_mask))

    return pixel_changes > pixel_threshold


def get_different_tiles_old(
    board_patches_start: np.ndarray, 
    board_patches_end: np.ndarray
) -> list[tuple]:
    """
    Given two boards, get the position where tiles have changed.
    The second board (`board_end`) should contain more tiles then
    the first one (`board_start`).

    Returns:
    --------
    new_tiles_idx: list[tuple]
        A list with tuples representing the position of the tiles that were added
        to the board. The list contains (row, colw) tuples.
    """
    new_tiles = []

    for i in range(NO_OF_TILES_PER_ROW):
        for j in range(NO_OF_TILES_PER_COL):            
            if check_different_tiles(board_patches_start[16 * i + j], board_patches_end[16 * i + j]):
                new_tiles.append((i, j))

    return new_tiles

def get_different_tiles(
    board_tile_start: list,
    board_tile_end: list,
) -> list[tuple]:
    """
    Given two boards, get the position where tiles have changed.
    The second board (`board_end`) should contain more tiles then
    the first one (`board_start`).

    Returns:
    --------
    new_tiles_idx: list[tuple]
        A list with tuples representing the position of the tiles that were added
        to the board. The list contains (row, colw) tuples.
    """
    new_tiles = []

    for i in range(NO_OF_TILES_PER_ROW):
        for j in range(NO_OF_TILES_PER_COL):
            if board_tile_start[i][j] == 0 and board_tile_end[i][j] == 1:
                new_tiles.append((i, j))

    return new_tiles