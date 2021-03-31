import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


data = pd.read_csv("data/titanic_train.csv")
data['Sex'] = [int(s == 'female') for s in data['Sex'].values]
data.Age = data.Age.fillna(data.Age.mean())

x = data[['Sex', 'Age', 'Pclass']].to_numpy()
y = data['Survived'].to_numpy()

estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=42)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print(accuracy_score(pred, y_test))
