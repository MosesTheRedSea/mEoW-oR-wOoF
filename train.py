import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils.lr_utils import load_dataset
from utils.helper import *

def model(X_train, Y_train, X_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    dim = X_train.shape[0]
    w = np.zeros((dim, 1))
    b = 0.0

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    w = params['w']
    b = params['b']
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        # print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

if __name__ == '__main__':

    train_X, train_Y, test_X, _, classes, test_ids = load_dataset(train_directory="/home/moses/Moses/Research/Machine-Learning/Data/Logistic/train", test_directory="/home/moses/Moses/Research/Machine-Learning/Data/Logistic/test", img_size=64)
    
    index = 25

    plt.imshow(train_X[index])
    
    print ("y = " + str(train_Y[:, index]) + ", it's a '" + classes[np.squeeze(train_Y[:, index])].decode("utf-8") +  "' picture.")

    m_train = train_X.shape[0]
    m_test = test_X.shape[0]
    num_px = train_X.shape[1]

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_set_x shape: " + str(train_X.shape))
    print ("train_set_y shape: " + str(train_Y.shape))
    print ("test_set_x shape: " + str(test_X.shape))
    
    # The Test Dataset has no labels
    #print ("test_set_y shape: " + str(Y.shape))s

    train_X_flatten = train_X.reshape(train_X.shape[0], -1).T
    test_X_flatten = test_X.reshape(test_X.shape[0], -1).T

    print ("train_set_x_flatten shape: " + str(train_X_flatten.shape))
    print ("train_set_y shape: " + str(train_Y.shape))
    print ("test_set_x_flatten shape: " + str(test_X_flatten.shape))
    #print ("test_set_y shape: " + str(test_set_y.shape))

    # Standardize the dataset
    train_X = train_X_flatten / 255
    test_X = test_X_flatten / 255

    logistic_regression_model = model(train_X, train_Y, test_X , num_iterations=2000, learning_rate=0.005, print_cost=True)

    costs = np.squeeze(logistic_regression_model['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
    plt.show()




    
    