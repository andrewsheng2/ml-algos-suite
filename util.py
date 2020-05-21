import numpy as np

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

def graph_linear(w, X_train, y_train, X_val, y_val, lmbd):
    plt.figure(num='Linear Regression', figsize=(8,5))
    plt.xlim(0, 150)
    plt.ylim(0, 450)

    plt.plot(X_train, y_train, "ro", label="Training Points")
    plt.plot(X_val, y_val, "bo", label="Validation Points")

    x = np.linspace(-50,150,200)
    y = w[1]*x + w[0]
    line_label = "Regression Line, lambda = " + str(lmbd)
    plt.plot(x,y, "k", label=line_label)

    plt.title("1D Linear Regression")
    plt.legend(loc="lower right")
    plt.show()

def graph_logistic(w, X_train, y_train, X_val, y_val, lmbd):
    plt.figure(num='Logistic Regression', figsize=(8,5))
    plt.xlim(0, 20)
    plt.ylim(-0.5, 1.5)

    plt.plot(X_train, y_train, "ro", label="Training Points")
    plt.plot(X_val, y_val, "bo", label="Validation Points")

    x = np.linspace(-5,25,100)
    z = w[1]*x + w[0]
    y = 1 / (1 + np.exp(-1*z))
    line_label = "Regression Curve, lambda = " + str(lmbd)
    plt.plot(x,y, "k", label=line_label)

    plt.title("1D Logistic Regression")
    plt.legend(loc="lower right")
    plt.show()

def graph_kernel(X_train, y_train, X_val, y_val, kernel, lmbd, predict_points):
    plt.figure(num='Kernel Regression', figsize=(8,5))
    plt.xlim(0, 150)
    plt.ylim(0, 450)

    plt.plot(X_train, y_train, "ro", label="Training Points")
    plt.plot(X_val, y_val, "bo", label="Validation Points")

    x = np.asarray([[i] for i in range(-50,150)])
    y = predict_points(x)
    line_label = "Prediction Curve with " + kernel + " Kernel, lambda = " + str(lmbd)
    plt.plot(x,y, "k", label=line_label)

    plt.title("1D Kernel Regression")
    plt.legend(loc="upper left")
    plt.show()

def show_decision_tree():
    pass

def show_kmeans():
    pass