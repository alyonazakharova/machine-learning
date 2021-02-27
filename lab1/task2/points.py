import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn import metrics

# var 4
# class -1
M_x1 = 13
M_x2 = 11
D = 3
N = 40

# class 1
M_1_x1 = 17
M_1_x2 = 12
D_1 = 2
N_1 = 60

x1 = np.random.normal(M_x1, D, N)
x2 = np.random.normal(M_x2, D, N)

x1_1 = np.random.normal(M_1_x1, D_1, N_1)
x2_1 = np.random.normal(M_1_x2, D_1, N_1)

plt.scatter(x1, x2, c='g')
plt.scatter(x1_1, x2_1, c='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

dataset = []
for i in range(N):
    dataset.append([x1[i], x2[i], -1])
for i in range(N_1):
    dataset.append([x1_1[i], x2_1[i], 1])

dataset = np.array(dataset)
features = dataset[:, :-1]
target = dataset[:, -1]

features_train, features_test, target_train, target_test =\
    train_test_split(features, target, test_size=0.15)

gnb = GaussianNB()
gnb.fit(features_train, target_train)

target_pred = gnb.predict(features_test)
print(accuracy_score(target_test, target_pred))
print(confusion_matrix(target_test, target_pred))

target_probabilities = gnb.predict_proba(features_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(target_test, target_probabilities)
plt.plot(fpr, tpr)
plt.title("ROC curve")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.show()

precision, recall, thresholds = precision_recall_curve(target_test, target_probabilities)
plt.plot(recall, precision)
plt.title("PR curve")
plt.xlabel("recall")
plt.ylabel("precision")
plt.show()
