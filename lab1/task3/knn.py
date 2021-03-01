import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../data/glass.csv')
features = df.iloc[:, 1:-1].to_numpy()
target = df['Type'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# a
err = []
for k in range(1, 101):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    err.append(accuracy_score(y_test, y_pred))

plt.plot(list(range(1, 101)), err)
plt.ylim(0, 1)
plt.xlabel('neighbors')
plt.ylabel('error')
plt.show()

# b
for metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
    knn = KNeighborsClassifier(n_neighbors=15, metric=metric)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(accuracy_score(y_test, y_pred))

# c
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(features, target)
print(knn.predict([[1.516, 11.7, 1.01, 1.19, 72.59,
                   0.43, 11.44, 0.02, 0.1]]))
