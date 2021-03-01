import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


df_train = pd.read_csv('../data/bank_scoring_train.csv', delimiter='\t')
x_train = df_train.iloc[:, 1:].to_numpy()
y_train = df_train['SeriousDlqin2yrs'].to_numpy()

df_test = pd.read_csv('../data/bank_scoring_test.csv', delimiter='\t')
x_test = df_test.iloc[:, 1:].to_numpy()
y_test = df_test['SeriousDlqin2yrs'].to_numpy()

print("Naive Bayes:")
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print(confusion_matrix(y_test, y_pred))

print("KNN:")
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(confusion_matrix(y_test, y_pred))

print("Decision tree:")
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
