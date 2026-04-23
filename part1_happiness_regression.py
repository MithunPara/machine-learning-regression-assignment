import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random
from Linear_Regression import data_preprocess, linear_regression

CSV_PATH = "datasets/gdp-vs-happiness.csv"

epochs = [100, 500, 1000, 1500, 2000]
learning_rates = [0.001, 0.00001, 0.0005, 0.03, 0.0001]
# epochs = [200, 500, 1000, 2000, 5000]
# learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
output = []


def gradient_descent(X, Y, eta, epoch):
    n = X.shape[0]
    beta = np.random.randn(2, 1) # random initialization for beta

    for _ in range(epoch):    
        gradients = 2/n * (X.T).dot(X.dot(beta) - Y)
        beta = beta - eta * gradients
    
    return beta

if __name__ == "__main__":

    """
    OUTPUT 1
    """

    happiness, gdp = data_preprocess(CSV_PATH)
    lr_gd = linear_regression(gdp, happiness)
    X, Y = lr_gd.preprocess()

    for eta in learning_rates:
        for epoch in epochs:
            beta = gradient_descent(X, Y, eta, epoch)
            error = X.dot(beta) - Y
            mse = np.mean(error**2)
            output.append((eta, epoch, beta, mse))

    # extract 8 different regression lines
    sorted_output = sorted(output, key = lambda x: x[3])
    regression_lines = random.sample(sorted_output, 8)

    # print the beta values, epochs and learning rates
    for eta, epoch, beta, mse in sorted_output:
        print(f"Learning Rate: {eta}, epoch: {epoch}, beta: {beta.ravel()}, MSE: {mse: .4f}")

    # access the 1st column (the 0th column is all 1's)
    X_ = X[...,1].ravel()

    fig, ax = plt.subplots()
    fig.set_size_inches((15,8))

    # display the X and Y points
    ax.scatter(X_,Y)

    for eta, epoch, beta, mse in regression_lines:
        Y_predict_gd = lr_gd.predict(X, beta)
        #display the line predicted by beta and X
        ax.plot(X_,Y_predict_gd,label=f"Learning rate: {eta}, epoch: {epoch}, beta: {beta.ravel()}, MSE: {mse: .4f}")

    #set the x-labels
    ax.set_xlabel("GDP per capita")

    #set the y-labels
    ax.set_ylabel("Happiness")

    #set the title
    ax.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

    ax.legend()

    #show the plot
    plt.show()

    """
    OUTPUT 2
    """

    lr_ols = linear_regression(gdp, happiness)
    best_eta, best_epoch, best_beta, best_mse = sorted_output[0] # All the attributes for the best gradient descent regression line in terms of MSE

    # don't need to preprocess the inputs again, already done in output 1 code

    #compute beta
    beta2 = lr_ols.train(X,Y)

    # use the computed beta for prediction
    Y_predict = lr_ols.predict(X,beta2)
    Y_predict_best_gd = lr_gd.predict(X, best_beta)

    print(f"Best GD Beta Learning Rate: {best_eta}, epoch: {best_epoch}, beta: {best_beta.ravel()}, MSE: {best_mse: .4f}")
    print(f"OLS beta: {beta2.ravel()}")

    # below code displays the predicted values

    # access the 1st column (the 0th column is all 1's)
    X_ = X[...,1].ravel()

    #set the plot and plot size
    fig2, ax2 = plt.subplots()
    fig2.set_size_inches((15,8))

    # display the X and Y points
    ax2.scatter(X_,Y)    

    ax2.plot(X_,Y_predict,label=f"OLS beta: {beta2.ravel()}")
    ax2.plot(X_,Y_predict_best_gd,label=f"Best GD Beta Learning rate: {best_eta}, epoch: {best_epoch}, beta: {best_beta.ravel()}, MSE: {best_mse: .4f}")

    #set the x-labels
    ax2.set_xlabel("GDP per capita")

    #set the y-labels
    ax2.set_ylabel("Happiness")

    #set the title
    ax2.set_title("Cantril Ladder Score vs GDP per capita of countries (2018)")

    ax2.legend()

    #show the plot
    plt.show()
