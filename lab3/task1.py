import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def get_metrics(x, y):
    print("Silhouette Coefficient:", "{:.5f}".format(metrics.silhouette_score(x, y, metric='euclidean')))
    print("Calinski-Harabasz Index:", "{:.5f}".format(metrics.calinski_harabasz_score(x, y)))
    print("Davies-Bouldin Index:", "{:.5f}".format(metrics.davies_bouldin_score(x, y)))


features = pd.read_csv("data/pluton.csv").to_numpy()

iterations = [1, 100, 1000]

print("---Non standardized data---")
for i in iterations:
    print("Max iter: ", i)
    model = KMeans(n_clusters=3, random_state=0, max_iter=i).fit(features)
    labels = model.labels_
    get_metrics(features, labels)
    print("\n")

scaler = StandardScaler()
features_std = scaler.fit_transform(features)

print("---Standardized data---")
for i in iterations:
    print("Max iter: ", i)
    model = KMeans(n_clusters=3, random_state=0, max_iter=i).fit(features_std)
    labels = model.labels_
    get_metrics(features_std, labels)
    print("\n")
