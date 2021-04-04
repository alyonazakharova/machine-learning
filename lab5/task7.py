import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


data = pd.read_csv("data/cars.csv")
print(data)

x = data.speed.values.reshape(-1, 1)
y = data.dist.values

# plt.plot(x, y)
# plt.xlabel('speed')
# plt.ylabel('distance')
# plt.show()
#
reg = LinearRegression().fit(x, y)
print(reg.predict(np.array(40).reshape(-1, 1)))
