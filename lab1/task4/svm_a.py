import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from machine_learning.lab1.task4.svm import read_data, make_meshgrid, plot_contours

x_train, x_test, y_train, y_test = read_data("../data/svmdata_a.txt", "../data/svmdata_a_test.txt")

clf = svm.SVC()
clf.fit(x_train, y_train)

print(clf.n_support_)

print("Train data:")
pred = clf.predict(x_train)
print(accuracy_score(y_train, pred))
print(confusion_matrix(y_train, pred))

print("Test data:")
pred = clf.predict(x_test)
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))


C = 1.0  # SVM regularization parameter
model = svm.SVC(kernel='linear', C=C)
clf = model.fit(x_train, y_train)
fig, ax = plt.subplots()
title = 'SVC with linear kernel'
X0, X1 = x_train[:, 0], x_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()
