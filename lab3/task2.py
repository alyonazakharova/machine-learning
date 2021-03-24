import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def visualize(x, y, title):
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title(title)
    plt.show()


def clusterize(x):
    for i in range(1, 5):
        model = KMeans(n_clusters=i, random_state=0).fit(x)
        y = model.predict(x)
        title = "K-Means, " + str(i) + " clusters"
        visualize(x, y, title)

    y = DBSCAN().fit_predict(x)
    visualize(x, y, "DBSCAN")

    for i in range(1, 5):
        y = AgglomerativeClustering(n_clusters=i).fit_predict(x)
        title = "Agglomerative Clustering, " + str(i) + " clusters"
        visualize(x, y, title)


scaler = StandardScaler()

x1 = pd.read_csv("data/clustering_1.csv", sep="\t").to_numpy()
x1_std = scaler.fit_transform(x1)
x2 = pd.read_csv("data/clustering_2.csv", sep="\t").to_numpy()
x2_std = scaler.fit_transform(x2)
x3 = pd.read_csv("data/clustering_3.csv", sep="\t").to_numpy()
x3_std = scaler.fit_transform(x3)

# clusterize(x1_std)
# clusterize(x2_std)
clusterize(x3_std)