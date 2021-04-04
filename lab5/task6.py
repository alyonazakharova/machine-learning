import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv("data/JohnsonJohnson.csv").to_numpy()
print(data)

y = data[:, 1]
years = np.arange(1960, 1981).reshape(-1, 1)

y1 = []
y2 = []
y3 = []
y4 = []
values = []

k = 0
year = 1960
while year <= 1980:
    y1.append(y[k])
    k += 1
    y2.append(y[k])
    k += 1
    y3.append(y[k])
    k += 1
    y4.append(y[k])
    k += 1
    values.append((y1[-1] + y2[-1] + y3[-1] + y4[-1]) / 4)
    year += 1

print(values)
reg = LinearRegression()

reg.fit(years, y1)
print("Q1:", reg.score(years, y1), reg.coef_[0])
print("2016 Q1:", reg.predict([[2016]])[0])
plt.plot(years, reg.predict(years), label='Q1')

reg.fit(years, y2)
print("Q2:", reg.score(years, y2), reg.coef_[0])
print("2016 Q2:", reg.predict([[2016]])[0])
plt.plot(years, reg.predict(years), label='Q2')

reg.fit(years, y3)
print("Q3:", reg.score(years, y3), reg.coef_[0])
print("2016 Q3:", reg.predict([[2016]])[0])
plt.plot(years, reg.predict(years), label='Q3')

reg.fit(years, y4)
print("Q4:", reg.score(years, y4), reg.coef_[0])
print("2016 Q4:", reg.predict([[2016]])[0])
plt.plot(years, reg.predict(years), label='Q4')

reg.fit(years, values)
print("Year:", reg.score(years, values), reg.coef_[0])
print("2016:", reg.predict([[2016]])[0])
plt.plot(years, reg.predict(years), label='Year')
plt.legend()
plt.show()

# plt.plot(years, y1, label="Q1")
# plt.plot(years, y2, label="Q2")
# plt.plot(years, y3, label="Q3")
# plt.plot(years, y4, label="Q4")
# plt.legend()
# plt.show()
