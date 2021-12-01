import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
from scipy.interpolate import make_interp_spline
from scipy.optimize import leastsq
import tensorflow as tf

CANNY_VALUE = 85
'''
图片中 49 像素 = 200 um
'''
UM_PER_PIXEL = 200 / 49


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


def remove_noise(l: list):
    step = 10
    max_ = max(l)
    min_ = min(l)
    hi = max_ - (max_ - min_) * 0.1
    lo = min_ + (max_ - min_) * 0.1
    for index, num in enumerate(l):
        if index - step >= 0 and index + step < len(l):
            nearby = l[index - step: index + step]
            average = sum(nearby) / len(nearby)
            if abs(num - average) > 30 or num > hi or num < lo:
                l[index] = average
    return l


def min_fit(l: list, title: str):
    def Fun(p, x):  # 定义拟合函数形式
        a0, a1, a2, a3 = p
        return a3 * x ** 3 + a2 * x ** 2 + a1 * x + a0

    def error(p, x, y):  # 拟合残差
        return Fun(p, x) - y

    x = np.array([i for i in range(len(l))])  # 创建时间序列
    p0 = np.array([1, 0.1, -0.01, 10])  # 拟合的初始参数设置
    para = leastsq(error, p0, args=(x, l))  # 进行拟合
    y_fitted = Fun(para[0], x)  # 画出拟合后的曲线\
    para_fit = para[0]
    if len(title) > 0:
        plt.plot(x, l, 'r', label='Original curve')
        plt.plot(x, y_fitted, '-b', label='Fitted curve')
        plt.title(title)
        plt.legend()
        plt.show()
        print(para_fit)
    return min(y_fitted)


def canny_func(img: np.array):
    gauss = cv.GaussianBlur(img, (3, 3), 1)
    canny = cv.Canny(gauss, CANNY_VALUE, CANNY_VALUE * 2.5)
    return canny


def cst_from_img(img: np.array, name: str):
    canny = canny_func(img)
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    points = []  # 边缘的集合
    for i in contours:
        for j in i:
            points.append(j[0])
    points.sort(key=lambda x: x[0])
    # cv.imshow("", canny)
    # cv.waitKey(0)  # 按任意键关闭图片
    delta_h = []  # 测量到的上沿
    now_x = points[0][0]
    group = []
    for i in points:
        if now_x == i[0]:
            group.append(i[1])
        else:
            high = max(group)
            low = min(group)
            if 3 < high - low:
                delta_h.append(high - low)
            group.clear()
            now_x = i[0]
            group.append(i[1])

    # 最小二乘法拟合
    return min_fit(delta_h, name) * UM_PER_PIXEL


if __name__ == '__main__':
    img = cv.imread("0000-0004L_2002.jpg", 0)
    img = img[:500, 500:]
    img = canny_func(img)
    # canny_show(img)
    print("result:", cst_from_img(img, "0000-0004L_2002.jpg"))
