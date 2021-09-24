import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
from scipy.interpolate import make_interp_spline

CANNY_VALUE = 98
'''
图片中 50 像素 = 200 um
'''
UM_PER_PIXEL = 4


def canny_show(img):
    '''
    Use a bar to adjust canny value and see the result.
    调整canny值，并查看结果
    :param img: image object or np.array
    :return: None
    '''
    gauss = cv.GaussianBlur(img, (3, 3), 1)

    def doCanny(x):
        position = cv.getTrackbarPos("CannyBar", "Canny")
        canny = cv.Canny(gauss, position, position * 2.5)
        cv.imshow("Canny", canny)

    cv.namedWindow("Canny")
    cv.createTrackbar("CannyBar", "Canny", 1, 180, doCanny)  # 目前认为133是较为合适的值
    cv.waitKey(0)


def remove_noise(l:list):
    ls = []
    step = 5
    for index, num in enumerate(l):
        if index - step >= 0 and index + step < len(l):
            nearby = l[index - step: index + step]
            average = sum(nearby) / len(nearby)
            if abs(l[index] - average) < 20:
                ls.append(l[index])
    return ls


if __name__ == '__main__':
    img = cv.imread("0000-0004L_2002.jpg", 0)
    img = img[:500, 500:]
    # original_img = copy.deepcopy(img)
    # canny_show(img)
    gauss = cv.GaussianBlur(img, (3, 3), 1)
    canny = cv.Canny(gauss, CANNY_VALUE, CANNY_VALUE * 2.5)
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    points = []  # 边缘的集合
    for i in contours:
        for j in i:
            points.append(j[0])
    points.sort(key=lambda x: x[0])
    cv.imshow("", canny)
    cv.waitKey(0)  # 按任意键关闭图片
    delta_h = []  # 测量到的厚度
    now_x = points[0][0]
    group = []
    for i in points:
        if now_x == i[0]:
            group.append(i[1])
        else:
            high = max(group)
            low = min(group)
            if 3 < high - low:
                delta_h.append(high-low)
            group.clear()
            now_x = i[0]
            group.append(i[1])
    delta_h = remove_noise(delta_h)

    print(delta_h)
    plt.scatter([i for i in range(len(delta_h))], delta_h)
    plt.show()

    # delta_h = delta_h[50:150]  # 只取中间
    delta_h.sort()
    r = sum(delta_h[10:40]) / 30
    r = r * UM_PER_PIXEL
    print("result:", r)
    print("0000-0004L	1	59	6	0.06	0	222	0	0	0	0	0	0	221")
