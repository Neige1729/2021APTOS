import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy

def reverse_color(image):
    height, width = image.shape
    print("width:%s,height:%s" % (width, height))

    for row in range(height):
        for line in range(width):
            pv = image[row, line]
            image[row, line] = 255 - pv


if __name__ == '__main__':
    img = cv.imread("example.jpg", 0)
    img = img[:500, 500:]
    original_img = copy.deepcopy(img)
    # hight, width = img.shape
    img_gauss = cv.GaussianBlur(img, (3, 3), 1)
    reverse_color(img_gauss)
    ones = np.ones_like(img_gauss)
    img_gauss = img_gauss - ones * 10
    lower = 0
    upper = 90
    mask = cv.inRange(img_gauss, lower, upper)
    plt.subplot(1, 2, 1)
    plt.title("original")
    plt.imshow(original_img, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("mask")
    plt.imshow(mask, cmap="gray")
    plt.show()
