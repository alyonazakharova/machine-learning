from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from machine_learning.lab1.task4.svm import read_data

x_train, x_test, y_train, y_test = read_data("data/svmdata_b.txt", "data/svmdata_b_test.txt")

# clf = svm.SVC()
clf = svm.SVC(kernel='linear', C=1000)
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
