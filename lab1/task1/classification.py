from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def classify(x, y):
    gnb = GaussianNB()

    train_size_arr = []
    acc_test_arr = []
    acc_train_arr = []

    train_size = 0.1
    for i in range(9):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
        gnb.fit(x_train, y_train)

        test_pred = gnb.predict(x_test)
        train_pred = gnb.predict(x_train)

        acc_test_arr.append(accuracy_score(y_test, test_pred))
        acc_train_arr.append(accuracy_score(y_train, train_pred))
        train_size_arr.append(train_size)

        train_size += 0.1

    return train_size_arr, acc_test_arr, acc_train_arr


def accuracy_plot(train_size, acc_test, acc_train, title):
    plt.plot(train_size, acc_test, label='test')
    plt.plot(train_size, acc_train, label='train')
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.xlabel('train size')
    plt.ylabel('accuracy')
    plt.title(title)
    plt.show()
