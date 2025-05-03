import os
import cv2 as cv
import numpy as np

def show_image(image: np.ndarray, name: str = "image") -> None:
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def read_template(path: list[str]) -> np.ndarray | None:
    template_path = os.path.join(*path)
    template_path = os.path.normpath(template_path)
    template = cv.imread(template_path)
    template = cv.resize(template, (100, 100))

    return template