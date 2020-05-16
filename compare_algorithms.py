import numpy as np
import time

import sys
import os

from linear_regression import linear_regression
# from kernel_regression import kernel_regression

from logistic_regression import logistic_gradient
from logistic_regression import logistic_newtons

# from decision_tree import learn_decision_tree

# from kmeans import kmeans
# from gmm import learn_gmm
# from neural_network import learn_nerual_network
# from recommender import learn_recommender

import util

#########################
#### PRINT FUNCTIONS ####
#########################

def print_welcome():
    f = open("texts/welcome.txt")
    print(f.read())

def print_linear_datasets():
    f = open("texts/linear_datasets.txt")
    print(f.read())

def print_logistic_datasets():
    f = open("texts/logistic_datasets.txt")
    print(f.read())

def clear():
    os.system('cls' if os.name=='nt' else 'clear')

#######################
#### LOAD DATASETS ####
#######################

def csv_to_ndarray(filename):
    with open(filename) as f:
        data_list = []
        for line in f:
            data_list.append([float(s) for s in line.split(',')])
    data_numpy = np.array(data_list)
    return data_numpy

def load_1D_regression():
    print("Loading data...")
    toy_1D_data = csv_to_ndarray('data/toy_1D_regression.csv')
    X_train = toy_1D_data[:45,1:]
    y_train = toy_1D_data[:45,0:1]
    X_val = toy_1D_data[45:,1:]
    y_val = toy_1D_data[45:,0:1]
    return (X_train, y_train, X_val, y_val)

def load_1D_classification():
    print("Loading data...")
    toy_1D_data = csv_to_ndarray('data/toy_1D_classification.csv')
    X_train = toy_1D_data[:40,1:]
    y_train = toy_1D_data[:40,0:1]
    X_val = toy_1D_data[40:,1:]
    y_val = toy_1D_data[40:,0:1]
    return (X_train, y_train, X_val, y_val)

def load_2D_classification():
    pass

def load_MNIST():
    print("Loading data...")
    train_data = csv_to_ndarray('data/mnist_train.csv')
    X_train = train_data[:,1:]
    y_train = train_data[:,0:1]
    val_data = csv_to_ndarray('data/mnist_val.csv')
    X_val = val_data[:,1:]
    y_val = val_data[:,0:1]
    return (X_train, y_train, X_val, y_val)

def load_ratings():
    pass

##########################
#### REQUESTS TO USER ####
##########################

def request_algorithm(num_algs):
    # ask user to input integer value of algorithm selection
    alg = None
    while alg == None:
        alg_input = input("Which algorithm do you want to run? Type 'exit' to close. ")
        try:
            if int(alg_input) in range(1,num_algs+1): alg = int(alg_input)
            else: print("Error: Not a valid selection")
        except:
            if alg_input == "exit": sys.exit()
            print("Error: Input value is not an integer")
    return alg

def request_dataset(num_sets):
    # ask user to input integer value of dataset selection
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
    # ask user to input positive float value for specified hyperparameter
    # possible hyperparameters: lambda for regularization, learning rate
    #          for gradient descent, number of iterations for gradient descent
    #          and newton's method`
    hyp = None
    while hyp == None:
        hyp_input = input("Please provide a positive value for "+name+": ")
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
    # ask user whether they want to visualize dataset, returns boolean
    while True:
        vis_input = input("Do you want to visualize the trained model? (Y/n) ")
        if vis_input.lower() == "y" or vis_input.lower() == "yes": return True
        elif vis_input.lower() == "n" or vis_input.lower() == "no": return False

def return_main():
    # ask user whether they want to return to main menu or exit program, doesn't return value
    req = None
    while req == None:
        req = input("Do you want to 'return' to the main menu or 'exit'? ")
        if req == "return" or req == "r":
            run()
        elif req == "exit" or req == "e":
            sys.exit()
        else:
            req = None

########################
#### RUN ALGORITHMS ####
########################

def run_linear(reg):
    print("Selected Linear Regression", end="")
    if reg: print(" with L2 Regularization.")
    else: print(".")
    print_linear_datasets()

    # load requested dataset
    dataset = request_dataset(2)
    if dataset == 1: X_train, y_train, X_val, y_val = load_1D_regression()
    if dataset == 2: X_train, y_train, X_val, y_val = load_MNIST()

    # request hyperparameters
    lmbd = request_hyperparameter("lambda") if reg else 0

    # start training and time
    print("Starting training...")
    start_time = time.time()
    (w, mse_train, mse_val) = linear_regression(X_train, y_train, X_val, y_val, lmbd)
    end_time = time.time()
    total = round(end_time-start_time, 4)

    # display results and request for visualization
    print("Finished training in", total, "seconds.")
    print("Training MSE: ", mse_train)
    print("Validation MSE: ", mse_val)
    if dataset == 1 and request_visualization():
        print("Displaying model...")
        util.graph_linear(w, X_train, y_train, X_val, y_val, lmbd)
    return_main()

def run_logistic(mode, reg):
    print("Selected Linear Regression", end="")
    print(" using "+mode, end="")
    if reg: print(" with L2 Regularization.")
    else: print(".")
    print_logistic_datasets()

    # load requested dataset
    dataset = request_dataset(1)
    if dataset == 1: X_train, y_train, X_val, y_val = load_1D_classification()

    # request hyperparameters
    lmbd = request_hyperparameter("lambda") if reg else 0
    alpha = request_hyperparameter("learning rate") if mode == "Gradient Descent" else 0
    num_iter = int(request_hyperparameter("number of iterations"))

    # start training and time
    print("Starting training...")
    start_time = time.time()
    if mode == "Gradient Descent":
        (w, obj_train, obj_val) = logistic_gradient(X_train, y_train, X_val, y_val, lmbd, alpha, num_iter)
    elif mode == "Newton's Method":
        (w, obj_train, obj_val) = logistic_newtons(X_train, y_train, X_val, y_val, lmbd, num_iter)
    end_time = time.time()
    total = round(end_time-start_time, 4)

    # display results and request for visualization
    print("Finished training in", total, "seconds.")
    print("Training Objective: ", obj_train)
    print("Validation Objective: ", obj_val)
    if dataset == 1 and request_visualization():
        print("Displaying model...")
        util.graph_logistic(w, X_train, y_train, X_val, y_val, lmbd)
    return_main()

###########################
#### MAIN RUN FUNCTION ####
###########################

def run():
    clear()
    print_welcome()
    alg = request_algorithm(10)
    clear()
    if alg == 1: run_linear(False)
    if alg == 2: run_linear(True)
    if alg == 7: run_logistic("Gradient Descent", False)
    if alg == 8: run_logistic("Newton's Method", False)
    if alg == 9: run_logistic("Gradient Descent", True)
    if alg == 10: run_logistic("Newton's Method", True)

if __name__ == "__main__":
    run()