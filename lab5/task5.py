import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv("data/eustock.csv")

data = data.to_numpy()
x1 = data[:, 0].reshape(-1, 1)
x2 = data[:, 1].reshape(-1, 1)
x3 = data[:, 2].reshape(-1, 1)
x4 = data[:, 3].reshape(-1, 1)

x = [i for i in range(len(data))]
x = np.array(x).reshape(-1, 1)

plt.plot(x, x1, label='DAX')
plt.plot(x, x2, label='SMI')
plt.plot(x, x3, label='CAC')
plt.plot(x, x4, label='FTSE')
plt.xticks([])
plt.yticks([])
plt.legend()
plt.show()

reg = LinearRegression()

print('DAX:')
reg.fit(x, x1)
print("Accuracy:", reg.score(x, x1))
print("Coefficient", reg.coef_[0])

print('SMI:')
reg.fit(x, x2)
print("Accuracy:", reg.score(x, x2))
print("Coefficient", reg.coef_[0])

print('CAC:')
reg.fit(x, x3)
print("Accuracy:", reg.score(x, x3))
print("Coefficient", reg.coef_[0])

print('FTSE:')
reg.fit(x, x4)
print("Accuracy:", reg.score(x, x4))
print("Coefficient", reg.coef_[0])

print("All:")
reg.fit(x, data)
print("Accuracy:", reg.score(x, data))
print("Coefficient", reg.coef_[0])


