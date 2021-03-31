import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv("data/glass.csv").to_numpy()
x = data[:, :-1]
y = data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2)

classifiers_number = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
acc_arr = []

# SVC(), KNeighborsClassifier(), DecisionTreeClassifier()
for n in classifiers_number:
    clf = BaggingClassifier(base_estimator=SVC(), n_estimators=n)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    accuracy = accuracy_score(pred, y_test)
    acc_arr.append(accuracy)
    print(accuracy)

plt.plot(classifiers_number, acc_arr)
plt.title("SVC")
plt.xlabel("estimators number")
plt.ylabel("accuracy")
plt.show()

print("average accuracy:", sum(acc_arr)/len(acc_arr))





