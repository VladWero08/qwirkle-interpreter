import cv2 as cv
import os

from src.board_extraction import get_board
from src.helpers import show_image

DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(DIR, "images")

BOARD_1_PATH = os.path.join(IMAGES_DIR, "board_1.jpg")
BOARD_2_PATH = os.path.join(IMAGES_DIR, "board_2.jpg")
BOARD_3_PATH = os.path.join(IMAGES_DIR, "board_3.jpg")
BOARD_PATHS = [BOARD_1_PATH, BOARD_2_PATH, BOARD_3_PATH]

for i in range(len(BOARD_PATHS)):
    board_img = cv.imread(BOARD_1_PATH)
    board = get_board(board_img)
    board_cropped_path = os.path.join(IMAGES_DIR, f"board_{i+1}_cropped.jpg")

    cv.imwrite(board_cropped_path, board)    