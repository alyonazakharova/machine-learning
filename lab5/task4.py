import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("data/longley.csv")
# print(data)

data = data.drop('Population', axis=1).to_numpy()
x = data[:, :-1]
y = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, shuffle=True)
reg = LinearRegression()
reg.fit(x_train, y_train)
print("Linear Regression:")
print("Test data error:", 1 - reg.score(x_test, y_test))
print("Train data error:", 1 - reg.score(x_train, y_train))


alphas = [10 ** (-3 + 0.2 * i) for i in range(26)]
# print(alphas)
test_err = []
train_err = []

for alpha in alphas:
    reg = Ridge(alpha=alpha)
    reg.fit(x_train, y_train)
    test_err.append(1 - reg.score(x_test, y_test))
    train_err.append(1 - reg.score(x_train, y_train))

plt.plot(alphas, test_err, 'r')
plt.plot(alphas, train_err, 'b')
plt.xlabel('alpha')
plt.ylabel('error')
plt.show()

print("Ridge:")
print("Test data error:", np.mean(test_err))
print("Train data error:", np.mean(train_err))
