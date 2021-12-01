from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras import layers
import matplotlib.pyplot as plt
from cv_main import canny_func, cst_from_img
import pandas as pd
import numpy as np
import os
import cv2 as cv
import tensorflow as tf
import tensorflow_hub as hub
import json

TRAIN_PATH = "D:\\Data\\ATOPS\\TrainingSet"
inception_resnet_v2_path = "pretrained_models\\inception_resnet_v2"


class Relative2dot5PercentAcc(tf.keras.metrics.Metric):
    def __init__(self, name="accuracy", **kwargs):
        super(Relative2dot5PercentAcc, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 在此处改代码
        # 该指标可以计算有多少样本被正确分类为属于给定类：
        values = (abs(y_true - y_pred) / y_true) <= 2.5e-2
        values = tf.cast(values, "float32")
        print('values', values)
        print('sample_weight', sample_weight)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.correct.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.correct

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.correct.assign(0.0)


def get_label():
    data_df = pd.read_csv("TrainingImputed.csv")
    IDs = data_df["patient ID"]
    preCST = data_df["preCST"]
    dict_id_precst = dict()
    for i in range(len(IDs)):
        dict_id_precst[IDs[i]] = preCST[i]
    return dict_id_precst


def custom_loss(y_actual, y_pred):
    print(y_pred, y_actual)
    return abs(y_actual - y_pred) / y_actual


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((500, 764, 1)))
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    model.add(normalization_layer)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="sigmoid"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation=None))
    model.compile(Adam(0.1), loss=custom_loss, metrics=[Relative2dot5PercentAcc()])
    model.summary()
    return model


if __name__ == '__main__':
    ls = os.listdir(TRAIN_PATH)
    d = get_label()
    x = []
    y = []
    for i in ls:
        side = ""
        imgs = os.listdir(TRAIN_PATH + "\\" + i)
        imgs_in_use = []
        for img in imgs:
            name_split = img.split('_')
            side = name_split[0][-1]
            if name_split[-1][0] == '2':
                imgs_in_use.append(img)
        rs = []
        for j in imgs_in_use:
            p = TRAIN_PATH + "/" + i + "/" + j
            img = cv.imread(p, 0)
            img = img[:500, 500:]
            # img = canny_func(img)
            x.append(img)
            if i[-1] not in ['L', 'R']:
                y.append(d[i + side])
            else:
                y.append(d[i])
    x = np.array(x)
    x = x.reshape((x.shape[0], 500, 764, 1))
    y = np.array(y)
    print("x", x.shape)
    print("y", y.shape)

    model = build_model()
    history = model.fit(x, y, epochs=20, validation_split=0.2)

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = history.epoch

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # correct = 0
    # for index, pic in enumerate(x):
    #     r = model.predict(np.array(pic))[0]
    #     if abs(r - y[index]) < y[index] * 2.5e-2:
    #         correct += 1
    #         print(f"I: Testing {index}... correct")
    #     print(f"I: Testing {index}... wrong")
    # print(f"Result: {len(x)} test in total, {correct} of them are correct, acc={correct/len(x)}")
