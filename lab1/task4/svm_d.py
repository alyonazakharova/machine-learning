from sklearn import svm

from machine_learning.lab1.task4.svm import read_data, visualize


x_train, x_test, y_train, y_test = read_data("data/svmdata_d.txt", "data/svmdata_d_test.txt")

C = 1.0  # SVM regularization parameter
# полиномиальное степеней 1-5, сигмоидальная функция, гауссово
poly_svc_1 = svm.SVC(kernel='poly', degree=1, C=C).fit(x_train, y_train)
poly_svc_2 = svm.SVC(kernel='poly', degree=2, C=C).fit(x_train, y_train)
poly_svc_3 = svm.SVC(kernel='poly', degree=3, C=C).fit(x_train, y_train)
poly_svc_4 = svm.SVC(kernel='poly', degree=4, C=C).fit(x_train, y_train)
poly_svc_5 = svm.SVC(kernel='poly', degree=5, C=C).fit(x_train, y_train)
sigmoid_svc = svm.SVC(kernel='sigmoid', C=C).fit(x_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', C=C).fit(x_train, y_train)

models = [poly_svc_1, poly_svc_2, poly_svc_3,
          poly_svc_4, poly_svc_5, sigmoid_svc, rbf_svc]

titles = ['Polynomial (degree 1) kernel',
          'Polynomial (degree 2) kernel',
          'Polynomial (degree 3) kernel',
          'Polynomial (degree 4) kernel',
          'Polynomial (degree 5) kernel',
          'Sigmoid kernel',
          'RBF kernel']

visualize(models, titles, x_train, y_train)
