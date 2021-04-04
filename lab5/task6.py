import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv("data/JohnsonJohnson.csv").to_numpy()

index = data[:, 0]
y = data[:, 1]

x = []
xQ1 = []
xQ2 = []
xQ3 = []
xQ4 = []
yQ1 = []
yQ2 = []
yQ3 = []
yQ4 = []

years = []
values = np.zeros(21)
year = 1960
k = 0
while year <= 1980:
    years.append(year)
    for i in range(4):
        values[year - 1960] += y[k]
        k += 1
    year += 1

years = np.array(years).reshape(-1, 1)
values = np.array(values).reshape(-1, 1)


for i in range(len(data)):
    if index[i].endswith('Q1'):
        x.append(float(index[i][:4]))
        xQ1.append(float(index[i][:4]))
        yQ1.append(y[i])
    elif index[i].endswith('Q2'):
        x.append(float(index[i][:4] + '.25'))
        xQ2.append(float(index[i][:4]))
        yQ2.append(y[i])
    elif index[i].endswith('Q3'):
        x.append(float(index[i][:4] + '.5'))
        xQ3.append(float(index[i][:4]))
        yQ3.append(y[i])
    elif index[i].endswith('Q4'):
        x.append(float(index[i][:4] + '.75'))
        xQ4.append(float(index[i][:4]))
        yQ4.append(y[i])
x = np.array(x).reshape(-1, 1)
xQ1 = np.array(xQ1).reshape(-1, 1)
xQ2 = np.array(xQ2).reshape(-1, 1)
xQ3 = np.array(xQ3).reshape(-1, 1)
xQ4 = np.array(xQ4).reshape(-1, 1)
yQ1 = np.array(yQ1).reshape(-1, 1)
yQ2 = np.array(yQ2).reshape(-1, 1)
yQ3 = np.array(yQ3).reshape(-1, 1)
yQ4 = np.array(yQ4).reshape(-1, 1)

reg = LinearRegression()

reg.fit(xQ1, yQ1)
print("Q1:", reg.score(xQ1, yQ1), reg.coef_[0])

reg.fit(xQ2, yQ2)
print("Q2:", reg.score(xQ2, yQ2), reg.coef_[0])

reg.fit(xQ3, yQ3)
print("Q3:", reg.score(xQ3, yQ3), reg.coef_[0])

reg.fit(xQ4, yQ4)
print("Q4:", reg.score(xQ4, yQ4), reg.coef_[0])

plt.plot(x, y)
plt.show()

print("2016 Q1:", reg.predict(np.array(2016.).reshape(-1, 1)))
print("2016 Q2:", reg.predict(np.array(2016.25).reshape(-1, 1)))
print("2016 Q3:", reg.predict(np.array(2016.5).reshape(-1, 1)))
print("2016 Q4:", reg.predict(np.array(2016.75).reshape(-1, 1)))

reg = LinearRegression()
reg.fit(values, years)
# reg.fit(np.array(years).reshape(-1, 1), np.array(values).reshape(-1, 1))
print("All:", reg.score(values, years), reg.coef_[0])
# print("All:", reg.score(np.array(years).reshape(-1, 1), np.array(values).reshape(-1, 1)), reg.coef_[0])
print("2016:", reg.predict(np.array(2016).reshape(-1, 1)))
