import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Dropout, BatchNormalization, Conv1D
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

col_x = ["gender", "age", "diagnosis", "preVA", "anti-VEGF"]
# col_y = ["preCST", "VA", "continue injection", "CST", "IRF", "SRF", "HRF"]
col_y = ["HRF"]

if __name__ == '__main__':
    df = pd.read_csv("TrainingImputed.csv")

    # load x
    x = df[col_x].values
    # x = np.array(x)
    print(x.shape)

    # load y
    y = []
    for i in df[col_y].values:
        if i == 0:
            y.append([1, 0])
        else:
            y.append([0, 1])
    y = np.array(y)
    print(y.shape)

    loss = "relu"

    model = Sequential()
    model.add(Input(shape=5))
    model.add(Dense(256, activation=loss))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation=loss))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=loss))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation=loss))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation=loss))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(2, activation="softmax"))

    model.compile(optimizer=Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=["accuracy"])

    model.summary()

    history = model.fit(x=x, y=y, batch_size=32, epochs=250, validation_split=0.2)

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

    model.save("model_v0")

    to_predict = pd.read_csv("PreliminaryValidationSet_Info.csv")
    p = model.predict(to_predict[col_x].values)
    r = np.argmax(p, axis=1)
    dataframe = pd.DataFrame({'HRF': r})
    dataframe.to_csv("temp.csv", index=False, sep=',')
    print(r)
