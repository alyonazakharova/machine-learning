import pandas as pd
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def analyze_depth(features_train, features_test,
                  target_train, target_test, depth):
    for i in range(1, depth):
        model = DecisionTreeClassifier(random_state=0, max_depth=i)
        model = model.fit(features_train, target_train)

        target_pred = model.predict(features_test)
        print("max depth =", i)
        print("accuracy =", accuracy_score(target_test, target_pred))


def analyze_criterion(features_train, features_test,
                      target_train, target_test):
    for criterion in ['gini', 'entropy']:
        model = DecisionTreeClassifier(random_state=0, criterion=criterion)
        model = model.fit(features_train, target_train)

        target_pred = model.predict(features_test)
        print("criterion:", criterion)
        print("accuracy =", accuracy_score(target_test, target_pred))


def visualize(model):
    dot_data = tree.export_graphviz(model, out_file=None)
    graph = graphviz.Source(dot_data, format="png")
    graph.render("decision_tree")


if __name__ == "__main__":
    df = pd.read_csv('../data/glass.csv')
    features = df.iloc[:, 1:-1].to_numpy()
    target = df['Type'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    clf = DecisionTreeClassifier(random_state=0)
    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))
    d = clf.get_depth()
    print(d)
    print("------------------------------------------")

    visualize(clf)

    analyze_depth(x_train, x_test, y_train, y_test, d)
    print("------------------------------------------")

    analyze_criterion(x_train, x_test, y_train, y_test)
