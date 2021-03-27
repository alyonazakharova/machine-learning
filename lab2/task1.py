import matplotlib.pyplot as plt
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split
from tensorflow import keras


def get_data(arr):
    features = arr[:, :-1]
    target = arr[:, -1]
    target[target == -1] = 0
    return features, target


def visualize_data(arr):
    class_1 = arr[arr[:, 2] == 0]
    class_2 = arr[arr[:, 2] == 1]

    plt.scatter(class_1[:, 0], class_1[:, 1])
    plt.scatter(class_2[:, 0], class_2[:, 1])
    plt.show()


dataset = pd.read_csv("data/nn_1.csv").to_numpy()
x, y = get_data(dataset)

# visualize_data(dataset)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

network = keras.Sequential([layers.Input(2),
                            layers.Dense(1, activation='sigmoid')]) # sigmoid, relu, tanh

network.compile(loss='binary_crossentropy',
                optimizer='SGD', #Adam, SGD
                metrics=['accuracy'])

epochs = [10, 100, 250, 500, 800, 1000]

for e in epochs:
    network.fit(x_train, y_train, epochs=e, verbose=0)
    loss, accuracy = network.evaluate(x_test, y_test, verbose=0)
    print("Epochs:", e, " loss:", "{:.5f}".format(loss), " accuracy:", "{:.5f}".format(accuracy))
