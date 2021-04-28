import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


data = pd.read_csv("data/nsw74psid1.csv").to_numpy()
# print(data)

x = data[:, :-1]
y = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg = DecisionTreeRegressor().fit(x_train, y_train)
print("Decision Tree:", reg.score(x_test, y_test))

reg = LinearRegression().fit(x_train, y_train)
print("Linear Regression:", reg.score(x_test, y_test))

reg = SVR().fit(x_train, y_train)
print("SVR:", reg.score(x_test, y_test))

reg = SVR(C=100).fit(x_train, y_train)
print("SVR, C=100:", reg.score(x_test, y_test))
print("SVR, C=100:", reg.score(x_test, y_test))
