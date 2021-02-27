import pandas as pd

from machine_learning.lab1.task1.classification import classify, accuracy_plot


if __name__ == "__main__":
    df = pd.read_csv("../data/tic_tac_toe.txt")
    df.columns = ['1', '2', '3', '4', '5',
                       '6', '7', '8', '9', 'res']

    features = df[['1', '2', '3', '4', '5',
                   '6', '7', '8', '9']].to_numpy()
    target = df['res'].to_numpy()

    f_dict = {'x': 1, 'o': 2, 'b': 3}
    f_mapped = [[f_dict[x] for x in i] for i in features]

    t_dict = {'positive': 1, 'negative': 2}
    t_mapped = [t_dict[x] for x in target]

    train_size, accuracy_test, accuracy_train = classify(f_mapped, t_mapped)
    accuracy_plot(train_size, accuracy_test, accuracy_train, 'Tic-tac-toe')
