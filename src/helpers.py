import cv2 as cv
import numpy as np

def show_image(image: np.ndarray, name: str = "image") -> None:
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()