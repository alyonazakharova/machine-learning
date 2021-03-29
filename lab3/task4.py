import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


x = pd.read_csv('data/votes.csv').to_numpy()

# for i in range(50):
#     print("i =", i, " -- ", x[i])


mean = np.nanmean(x, axis=0)
indices = np.where(np.isnan(x))
x[indices] = np.take(mean, indices[1])

i = 0
arr = []

for row in x:
    arr.append([i, sum(row) / 31])
    # print("i =", i, " -- ", arr[i])
    i += 1


arr = sorted(arr, key=lambda a: a[1])

for el in arr:
    print(f"State {el[0]}: {el[1]}")

model = AgglomerativeClustering(n_clusters=2).fit(x)
children = model.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0] + 2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
dendrogram(linkage_matrix)
plt.show()
