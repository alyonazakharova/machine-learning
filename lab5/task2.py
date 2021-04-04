import pandas as pd
from sklearn.linear_model import LinearRegression
import itertools


def RSS(pred, real):
    return sum((pred - real) ** 2)


data = pd.read_csv("data/reglab.txt", delimiter='\t')

y = data.y.values
x = data.drop('y', axis=1)

reg = LinearRegression()

reg = reg.fit(x, y)
y_pred = reg.predict(x)
print("RSS with all features:", RSS(y_pred, y))

all_features = ['x1', 'x2', 'x3', 'x4']
two_features = list(itertools.combinations(all_features, r=2))
three_features = list(itertools.combinations(all_features, r=3))

for i in all_features:
    x_reduced = x.drop(i, axis=1)
    reg = reg.fit(x_reduced, y)
    y_pred = reg.predict(x_reduced)
    print(f'RSS without {i}:', RSS(y_pred, y))

for i in two_features:
    x_reduced = x.drop(i[0], axis=1).drop(i[1], axis=1)
    reg = reg.fit(x_reduced, y)
    y_pred = reg.predict(x_reduced)
    print(f'RSS without {i[0], i[1]}:', RSS(y_pred, y))

for i in three_features:
    x_reduced = x.drop(i[0], axis=1).drop(i[1], axis=1).drop(i[2], axis=1)
    reg = reg.fit(x_reduced, y)
    y_pred = reg.predict(x_reduced)
    print(f'RSS without {i[0], i[1], i[2]}:', RSS(y_pred, y))






