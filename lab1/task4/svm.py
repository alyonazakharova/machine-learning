import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(train_data, test_data: str):
    df = pd.read_csv(train_data, delimiter="\t")
    features = df[['X1', 'X2']].to_numpy()
    target = df['Colors'].to_numpy()

    colors_dict = {'red': 0, 'green': 1}
    target_mapped = [colors_dict[i] for i in target]

    df = pd.read_csv(test_data, delimiter="\t")
    features_test = df[['X1', 'X2']].to_numpy()
    target_test = df['Colors'].to_numpy()

    target_test_mapped = [colors_dict[i] for i in target_test]

    return features, features_test, target_mapped, target_test_mapped


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def visualize(models, titles, x, y):
    for clf, title in zip(models, titles):
        fig, ax = plt.subplots()
        X0, X1 = x[:, 0], x[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        plt.show()