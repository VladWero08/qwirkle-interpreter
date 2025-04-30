import cv2 as cv

from src.helpers import show_image
from src.board_extraction import get_board
from src.tiles_extraction import get_patches

if __name__ == "__main__":
    image = cv.imread("./playground/board-perspective.jpg")
    board = get_board(image)
    show_image(board)
    patches = get_patches(board)