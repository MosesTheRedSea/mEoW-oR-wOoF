import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils.lr_utils import load_dataset
from utils.helper import *

def model(X_train, Y_train, X_test, num_iterations=2000, learning_rate=0.009, print_cost=False):

    dim = X_train.shape[0]
    w = np.zeros((dim, 1))
    b = 0.0

    params, grads, costs, accuracies = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    w = params['w']
    b = params['b']
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        # print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "accuracies": accuracies,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def save_run_outputs(outputs_directory, loss, accuracy):
    os.makedirs(output_directory, exist_ok=True)
    run_nums = []
    for name in os.listdir(output_directory):
        if of.path.isdir(os.path.join(output_directory, name)) and re.fullmatch(r"\d+", name):
             run_nums.append(int(name))
    next_run = max(run_nums, default=0) + 1
    run_dir = os.path.join(base, str(next_run))
    os.makedirs(run_dir)
    return run_dir

def save_loss_plot(loss, run_dir):
    plt.figure()
    plt.plot(loss)
    plt.ylabel("Loss")
    plt.xlabel("Iterations (per 100)")
    plt.title("Training Loss")
    run_number = int(os.path.basename(run_dir))
    save_path = os.path.join(run_dir, f"loss_run_{run_number}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot to {save_path}")

def save_accuracy_plot(train_acc, run_dir):
    plt.figure()
    plt.plot(train_acc)
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.title("Training Accuracy")
    run_number = int(os.path.basename(run_dir))
    save_path = os.path.join(run_dir, f"accuracy_run_{run_number}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy plot to {save_path}")

if __name__ == '__main__':

    train_X, train_Y, test_X, _, classes, test_ids = load_dataset(train_directory="/home/hice1/madewolu9/scratch/madewolu9/Data/train", test_directory="/home/hice1/madewolu9/scratch/madewolu9/Data/test", img_size=64)
    
    index = 25

    plt.imshow(train_X[index])
    
    print ("\n y = " + str(train_Y[:, index]) + ", it's a '" + classes[np.squeeze(train_Y[:, index])].decode("utf-8") +  "' picture.")

    m_train = train_X.shape[0]
    m_test = test_X.shape[0]
    num_px = train_X.shape[1]

    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Height/Width of each image: num_px = " + str(num_px))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3")

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

    logistic_regression_model = model(train_X, train_Y, test_X, num_iterations=10000, learning_rate=0.005, print_cost=True)

    costs = np.squeeze(logistic_regression_model['costs'])
    accuracies = np.squeeze(logistic_regression_model['accuracies'])

    run_dir = save_run_outputs("outputs/runs")
    save_loss_plot(costs, run_dir)
    save_accuracy_plot(accuracies, run_dir)
    print("Saving run to:", run_dir)

    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
    # plt.show()




    
    