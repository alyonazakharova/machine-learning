import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/reglab1.txt", delimiter='\t')
# print(data)

print("z(x, y):")
features = data[['x', 'y']].to_numpy()
target = data['z'].to_numpy()

reg = LinearRegression().fit(features, target)
print(reg.score(features, target))

print("y(x, z):")
features = data[['x', 'z']].to_numpy()
target = data['y'].to_numpy()

reg = LinearRegression().fit(features, target)
print(reg.score(features, target))

print("x(y, z):")
features = data[['y', 'z']].to_numpy()
target = data['x'].to_numpy()

reg = LinearRegression().fit(features, target)
print(reg.score(features, target))
