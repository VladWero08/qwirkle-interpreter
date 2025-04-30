import cv2 as cv
import numpy as np

from src.helpers import show_image

NO_OF_TILES_PER_ROW = 16
NO_OF_TILES_PER_COL = 16

def get_patches(
    image: np.ndarray, 
    width: int = 1600,
    height: int = 1600,
    padding: int = 10,    
) -> np.ndarray:
    # compute the patch width and height
    p_height = height // NO_OF_TILES_PER_ROW
    p_width = width // NO_OF_TILES_PER_COL
    patches = []

    for i in range(NO_OF_TILES_PER_ROW):
        for j in range(NO_OF_TILES_PER_COL):
            # compute the corners of the patch
            row_start = i * p_height 
            row_end = (i + 1) * p_height
            col_start = j * p_width 
            col_end = (j + 1) * p_width 

            patch = image[row_start:row_end, col_start:col_end]
            patches.append(patch)

    return np.array(patch)