import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR

data = pd.read_csv("data/svmdata6.txt", delimiter='\t')
# print(data)

x = data.X.values.reshape(-1, 1)
y = data.Y.values

eps = 0.1
eps_arr = []
err_arr = []
while eps < 1:
    eps_arr.append(eps)
    reg = SVR(kernel='rbf', C=1, epsilon=eps).fit(x, y)
    err = reg.score(x, y)
    err_arr.append(err)
    print(err)
    eps += 0.1

plt.plot(eps_arr, err_arr)
plt.xlabel('epsilon')
plt.ylabel('error')
plt.show()
