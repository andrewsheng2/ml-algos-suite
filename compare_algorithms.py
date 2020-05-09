import numpy as np
import sys
import time

from linear_regression import linear_regression
from logistic_regression import gradient_descent
from logistic_regression import newtons_method
from kernel_regression import kernel_regression

from decision_tree import learn_decision_tree
from kmeans import kmeans

from gmm import learn_gmm
from neural_network import learn_nerual_network
from recommender import learn_recommender

import util

def welcome():
    f = open("README.md")
    print(f.read())

def csv_to_ndarray(filename):
    with open(filename) as f:
        data_list = []
        for line in f:
            data_list.append([float(s) for s in line.split(',')])
    data_numpy = np.array(data_list)
    return data_numpy

def load_1D_regression():
    toy_1D_data = csv_to_ndarray('data/toy_1D_regression.csv')
    X_train = toy_1D_data[:45,1:]
    y_train = toy_1D_data[:45,0:1]
    X_val = toy_1D_data[45:,1:]
    y_val = toy_1D_data[45:,0:1]
    return (X_train, y_train, X_val, y_val)

def load_1D_classification():
    pass

def load_2D_classification():
    pass

def load_MNIST():
    pass

def load_ratings():
    pass

def request_algorithm():
    alg = None
    while alg == None:
        alg_input = input("Which algorithm do you want to run? Type 'exit' to close. ")
        try:
            if int(alg_input) in range(1,16): alg = int(alg_input)
            else: print("Error: Not a valid selection")
        except:
            if alg_input == "exit": sys.exit()
            print("Error: Input value is not an integer")
    return alg

def request_dataset(num_sets):
    dataset = None
    while dataset == None:
        set_input = input("Which dataset do you want to use? ")
        try:
            if int(set_input) in range(1,num_sets+1): dataset = int(set_input)
            else: print("Error: Not a valid selection")
        except:
            print("Error: Input value is not an integer")
    return dataset

def request_hyperparameter(name):
    hyp = None
    while hyp == None:
        hyp_input = input("Please provide a positive "+name+" value: ")
        try:
            if float(hyp_input) <= 0: print("Error: Inpupt value is not positive")
            else: hyp = float(hyp_input)
        except:
            print("Error: Input value is not a number")
        if hyp == float('inf'):
            print("Error: Input value cannot be infinity")
            hyp = None
    return hyp

def request_visualization():
    while True:
        vis_input = input("Do you want to visualize the trained model? (Y/n) ")
        if vis_input.lower() == "y" or vis_input.lower() == "yes": return True
        elif vis_input.lower() == "n" or vis_input.lower() == "no": return False

def return_main():
    req = None
    while req == None:
        req = input("Do you want to 'return' to the main menu or 'exit'? ")
        if req == "return":
            print("\n")
            run()
        elif req == "exit":
            sys.exit()
        else:
            req = None

def run_linear(reg):
    print("\nSelected Linear Regression", end="")
    if reg: print(" with L2 Regularization.")
    else: print(".")
    dataset = request_dataset(1)
    if dataset == 1: X_train, y_train, X_val, y_val = load_1D_regression()
    lmbd = request_hyperparameter("lambda") if reg else 0
    print("Starting training...")
    start_time = time.time()
    (w, mse_train, mse_val) = linear_regression(X_train, y_train, X_val, y_val, lmbd)
    end_time = time.time()
    total = round(end_time-start_time, 4)
    print("Finished training in", total, "seconds.")
    print("Training MSE: ", mse_train)
    print("Validation MSE: ", mse_val)
    if dataset == 1 and request_visualization():
        print("Displaying model...")
        util.graph_linear(w, X_train, y_train, X_val, y_val, lmbd)
    return_main()

def run_logistic_grad(reg):
    pass

def run_logistic_newton(reg):
    pass

def run_kernel():
    pass

def run():
    welcome()
    alg = request_algorithm()
    if alg == 1: run_linear(False)
    elif alg == 2: run_linear(True)

if __name__ == "__main__":
    run()