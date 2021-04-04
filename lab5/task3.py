import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = pd.read_csv("data/cygage.txt", delimiter='\t').to_numpy()

y = data[:, 0]
x = data[:, 1].reshape(-1, 1)
weights = data[:, 2]

plt.plot(x, y, 'o')

reg = LinearRegression()
reg.fit(x, y, sample_weight=weights)
print(reg.score(x, y, sample_weight=weights))
plt.plot(x, reg.predict(x), 'b', label='with weights')

reg.fit(x, y)
print(reg.score(x, y))
plt.plot(x, reg.predict(x), 'r', label='without weights')
plt.legend()
plt.show()
