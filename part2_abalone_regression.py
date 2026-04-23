import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import random
from Linear_Regression import data_preprocess, linear_regression

CSV_PATH = "datasets/training_data.csv"

FEATURE_TYPES = {
    "Length": "linear",
    "Diameter": "linear",
    "Height": "linear",
    "Whole_weight": "polynomial",
    "Shucked_weight": "polynomial",
    "Viscera_weight": "polynomial",
    "Shell_weight": "polynomial"
}

def process_data(csv_path=CSV_PATH):
    data = pd.read_csv(csv_path)

    # First column is not named so remove it, not sure if this will be the case for the other csv as well
    # so leave in if condition
    if data.columns[0] == "":
        data = data.drop(columns=[data.columns[0]])
    
    length, diameter, height, whole_weight = [], [], [], []
    shucked_weight, viscera_weight, shell_weight, rings = [], [], [], []

    for row in data.iterrows():
        if (pd.notna(row[1]['Length']) and pd.notna(row[1]['Diameter']) and pd.notna(row[1]['Height']) and pd.notna(row[1]['Whole_weight']) and
            pd.notna(row[1]['Shucked_weight']) and pd.notna(row[1]['Viscera_weight']) and pd.notna(row[1]['Shell_weight']) and pd.notna(row[1]['Rings'])):
                length.append(row[1]['Length'])
                diameter.append(row[1]['Diameter'])
                height.append(row[1]['Height'])
                whole_weight.append(row[1]['Whole_weight'])
                shucked_weight.append(row[1]['Shucked_weight'])
                viscera_weight.append(row[1]['Viscera_weight'])
                shell_weight.append(row[1]['Shell_weight'])
                rings.append(row[1]['Rings'])
    
    features = {
         "Length": length,
         "Diameter": diameter,
         "Height": height,
         "Whole_weight": whole_weight,
         "Shucked_weight": shucked_weight,
         "Viscera_weight": viscera_weight,
         "Shell_weight": shell_weight
    }

    return features, rings

def individual_feature_plots(features, rings):
    figs, axs = plt.subplots(2, 4, figsize=(18, 11))
    axs = axs.ravel()

    for i, feature in enumerate(features):
        ax = axs[i] 
        feat = features[feature] # Get the list of values for the feature
        lr = linear_regression(feat, rings)
        X, Y = lr.preprocess()
        beta = lr.train(X, Y)
        Y_predict = lr.predict(X, beta)

        X_ = X[...,1].ravel()
        ax.scatter(X_,Y)
        ax.plot(X_,Y_predict,label=f"OLS Line",color='r')
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel("Rings")
        ax.set_title(f"{feature} vs Rings")
        ax.legend()
    
    plt.show()

# Set training size to 2000 to represent the split in 2000 and 577 size sample sets
# Took the first 2000 as the sample set, unsure of the exact samples provided and whether they were ordered though
# Adjust training_size as needed depending on number of samples used for training versus testing
def split_training_testing(feature_values, testing_values, training_size=2000):
    X_train = feature_values[:training_size]
    Y_train = testing_values[:training_size]

    X_test = feature_values[training_size:]
    Y_test = testing_values[training_size:]

    return X_train, Y_train, X_test, Y_test

def training_testing_output(features, rings):
    for feature, feature_values in features.items():
        X_train_pre, Y_train_pre, X_test_pre, Y_test_pre = split_training_testing(feature_values, rings)

        feature_type = FEATURE_TYPES.get(feature)

        if feature_type == "linear":
            lr = linear_regression(X_train_pre, Y_train_pre)
            X_train, Y_train = lr.preprocess()
            beta = lr.train(X_train, Y_train)
            Y_predict = lr.predict(X_train, beta)

            hmean = np.mean(np.array(X_train_pre))
            hstd  = np.std(np.array(X_train_pre))
            x_test_std = (np.array(X_test_pre) - hmean) / hstd
            X_test = np.column_stack((np.ones(len(x_test_std)), x_test_std))

            gmean = np.mean(np.array(Y_train_pre))
            gstd  = np.std(np.array(Y_train_pre))
            Y_test = ((np.array(Y_test_pre) - gmean) / gstd).reshape(-1, 1)

            Y_predict_test = X_test.dot(beta)

        else:
            pr = polynomial_regression(X_train_pre, Y_train_pre)
            X_train, Y_train = pr.preprocess()
            beta = pr.train(X_train, Y_train)
            Y_predict = pr.predict(X_train, beta)

            hmean = np.mean(np.array(X_train_pre))
            hstd  = np.std(np.array(X_train_pre))
            x_test_std = (np.array(X_test_pre) - hmean) / hstd

            cols_test = [np.ones(len(x_test_std))]
            cols_test += [x_test_std**p for p in range(1, pr.degree + 1)]
            X_test = np.column_stack(cols_test)

            gmean = np.mean(np.array(Y_train_pre))
            gstd  = np.std(np.array(Y_train_pre))
            Y_test = ((np.array(Y_test_pre) - gmean) / gstd).reshape(-1, 1)

            Y_predict_test = X_test.dot(beta)

        mse_train = np.mean((Y_predict.ravel() - Y_train.ravel())**2)
        mse_test = np.mean((Y_predict_test.ravel() - Y_test.ravel())**2)

        print(f"Feature: {feature}, Feature Type: {feature_type}, beta: {beta.ravel()}, Training MSE: {mse_train: .4f}, Testing MSE: {mse_test: .4f}")

        # Output the training/testing data graphs for each feature one at a time
        figs, axs = plt.subplots(1, 2, figsize=(16, 9))
        ax_train, ax_test = axs[0], axs[1]

        X_ = X_train[...,1].ravel()
        ordered_train = np.argsort(X_)
        ax_train.scatter(X_, Y_train)
        ax_train.plot(X_[ordered_train],np.asarray(Y_predict).ravel()[ordered_train],label=f"OLS Line, beta: {beta.ravel()}, Training MSE: {mse_train: .4f}",color='r')
        ax_train.set_xlabel(f"{feature}")
        ax_train.set_ylabel("Rings")
        ax_train.set_title(f"{feature} vs Rings, Training Data, {feature_type}, OLS and MSE")
        ax_train.legend()

        X_2 = X_test[...,1].ravel()
        ordered_test = np.argsort(X_2)
        ax_test.scatter(X_2, Y_test)
        ax_test.plot(X_2[ordered_test],np.asarray(Y_predict_test).ravel()[ordered_test],label=f"OLS Line, beta: {beta.ravel()}, Testing MSE: {mse_test: .4f}",color='r')
        ax_test.set_xlabel(f"{feature}")
        ax_test.set_ylabel("Rings")
        ax_test.set_title(f"{feature} vs Rings, Testing Data, {feature_type}, OLS and MSE")
        ax_test.legend()

        plt.show()
        plt.close(figs)

# Set up polynomial regression class for datasets showing a polynomial relationship
class polynomial_regression():
 
    # Set default degree to 2, will stick to 2 for the polynomial plots
    def __init__(self,x_:list,y_:list, degree: int=2) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)
        self.degree = degree

    def preprocess(self,):

        #normalize the values
        hmean = np.mean(self.input)
        hstd = np.std(self.input)
        x_train = (self.input - hmean)/hstd

        #set up the polynomial feature matrix
        cols = [np.ones(len(x_train))]
        cols += [x_train**p for p in range(1, self.degree + 1)]
        X = np.column_stack(cols)

        #normalize the values
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean)/gstd

        #arrange in matrix format
        Y = (np.column_stack(y_train)).T

        return X, Y

    def train(self, X, Y):
        #compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def predict(self, X_test,beta):
        #predict using beta
        Y_hat = X_test*beta.T
        return np.sum(Y_hat,axis=1)
     
if __name__ == "__main__":
    features, rings = process_data(CSV_PATH)
    individual_feature_plots(features, rings)
    training_testing_output(features, rings)

