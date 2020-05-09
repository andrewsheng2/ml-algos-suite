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
    y1 = w[1]*x + w[0]
    line_label = "Regression Line, lambda = " + str(lmbd)
    plt.plot(x,y1, "k", label=line_label)

    plt.title("Sweden Auto Insurance")
    plt.legend(loc="lower right")
    plt.show()

def graph_logistic():
    pass

def show_decision_tree():
    pass

def show_kmeans():
    pass