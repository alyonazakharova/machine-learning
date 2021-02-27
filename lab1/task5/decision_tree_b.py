import pandas as pd
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from decision_tree_a import analyze_depth, analyze_criterion


df = pd.read_csv('../data/spam7.csv')
features = (df.iloc[:, 1:-1]).to_numpy()
target = df['yesno'].to_numpy()

target_dict = {'y': 0, 'n': 1}
target_mapped = [target_dict[i] for i in target]

x_train, x_test, y_train, y_test = train_test_split(features, target_mapped, test_size=0.2)

clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
depth = clf.get_depth()
print(depth)
