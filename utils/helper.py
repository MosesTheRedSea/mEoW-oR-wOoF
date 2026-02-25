import copy
import math
import numpy as np
from tqdm.auto import trange

def sigmoid(z):
    return 1/ (1+np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid((np.dot(w.T, X) + b))
    # cost = np.dot((-1/m), np.sum(np.dot(Y, np.log(A)) + np.dot((1 - Y), np.log(1 - A))))
    cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
    dw = (1/m) * np.dot(X, (A - Y).T)    
    db = (1/m) * np.sum(A - Y)
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    accuracies = []

    for i in trange(num_iterations, desc="Training", unit="iter"):
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Predictions for Accuracy
        A = sigmoid(np.dot(w.T, X) + b)
        preds = (A > 0.5).astype(int)

        accuracy = 100 - np.mean(np.abs(preds - Y)) * 100

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            accuracies.append(accuracy)
            # Print the cost every 100 training iterations
            if print_cost:
                print(f"Iter {i:5d} | Loss: {cost:.4f} | Acc: {accuracy:.2f}%")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs, accuracies

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    return Y_prediction