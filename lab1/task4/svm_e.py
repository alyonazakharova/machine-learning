from sklearn import svm

from machine_learning.lab1.task4.svm import read_data, visualize


x_train, x_test, y_train, y_test = read_data("../data/svmdata_e.txt", "../data/svmdata_e_test.txt")

gamma = 0.2
poly_svc_1 = svm.SVC(kernel='poly', gamma=gamma).fit(x_train, y_train)
poly_svc_2 = svm.SVC(kernel='poly', degree=2, gamma=gamma).fit(x_train, y_train)
poly_svc_3 = svm.SVC(kernel='poly', degree=3, gamma=gamma).fit(x_train, y_train)
poly_svc_4 = svm.SVC(kernel='poly', degree=4, gamma=gamma).fit(x_train, y_train)
poly_svc_5 = svm.SVC(kernel='poly', degree=5, gamma=gamma).fit(x_train, y_train)
sigmoid_svc = svm.SVC(kernel='sigmoid', gamma=gamma).fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=gamma).fit(x_train, y_train)

models = [poly_svc_1,
          poly_svc_2,
          poly_svc_3,
          poly_svc_4,
          poly_svc_5,
          sigmoid_svc,
          rbf_svc]

titles = ['Polynomial (degree 1) kernel, gamma = ' + str(gamma),
          'Polynomial (degree 2) kernel, gamma = ' + str(gamma),
          'Polynomial (degree 3) kernel, gamma = ' + str(gamma),
          'Polynomial (degree 4) kernel, gamma = ' + str(gamma),
          'Polynomial (degree 5) kernel, gamma = ' + str(gamma),
          'Sigmoid kernel, gamma = ' + str(gamma),
          'RBF kernel, gamma = ' + str(gamma)]

visualize(models, titles, x_train, y_train)
