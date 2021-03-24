import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow import keras

from task1 import get_data

dataset = pd.read_csv("data/nn_1.csv").to_numpy()
x, y = get_data(dataset)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)


network = keras.Sequential([layers.Input(2),
                            layers.Dense(10, activation='relu'),
                            layers.Dense(1, activation='sigmoid')])

network.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])

epochs = [10, 100, 250, 500, 800, 1000]

for e in epochs:
    network.fit(x_train, y_train, epochs=e, verbose=0)
    loss, accuracy = network.evaluate(x_test, y_test, verbose=0)
    print("Epochs:", e, " loss:", "{:.5f}".format(loss), " accuracy:", "{:.5f}".format(accuracy))
    