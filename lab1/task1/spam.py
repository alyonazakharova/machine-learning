import pandas as pd
from machine_learning.lab1.task1.classification import classify, accuracy_plot

df = pd.read_csv('../data/spam.csv')
features = (df.iloc[:, 1:-1]).to_numpy()
target = df['type'].to_numpy()

spam_dict = {'spam': 0, 'nonspam': 1}
spam_mapped = [spam_dict[i] for i in target]

train_size, accuracy_test, accuracy_train = classify(features, spam_mapped)
accuracy_plot(train_size, accuracy_test, accuracy_train, 'Spam')
