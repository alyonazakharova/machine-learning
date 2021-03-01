import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from decision_tree_a import visualize

df = pd.read_csv('../data/spam7.csv')
features = (df.iloc[:, 1:-1]).to_numpy()
target = df['yesno'].to_numpy()

target_dict = {'n': 0, 'y': 1}
target_mapped = [target_dict[i] for i in target]


x_train, x_test, y_train, y_test = train_test_split(features, target_mapped, test_size=0.2)

clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

clf = DecisionTreeClassifier(max_depth=10, criterion='entropy')
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

clf = DecisionTreeClassifier(max_depth=10, criterion='entropy', class_weight={0: 1, 1: 0.05})
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

visualize(clf)
