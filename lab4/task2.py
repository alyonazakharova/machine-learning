import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("data/vehicle.csv").to_numpy()
x = data[:, :-1]
y = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

classifiers_number = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
acc_arr = []

# SVC(), KNeighborsClassifier(), DecisionTreeClassifier()
for n in classifiers_number:
    clf = AdaBoostClassifier(base_estimator=SVC(kernel='linear'), algorithm='SAMME', n_estimators=n, random_state=0)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    accuracy = accuracy_score(pred, y_test)
    acc_arr.append(accuracy)
    print(accuracy)

plt.plot(classifiers_number, acc_arr)
plt.title("SVC with linear kernel")
plt.xlabel("estimators number")
plt.ylabel("accuracy")
plt.show()

print("average accuracy:", sum(acc_arr)/len(acc_arr))
