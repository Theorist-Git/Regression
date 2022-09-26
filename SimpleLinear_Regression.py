"""
Copyright (C) 2021 Mayank Vats
See license.txt
"""

# Implementation of Uni-variate Linear Regression

import numpy as np
from random import randrange
from matplotlib import pyplot as plt


class SimpleLinearRegression:

    def __init__(self):
        self.slope = None
        self.intercept = None

    @staticmethod
    def create_dataset(n, variance, correlation, step=2):
        """
        Creates a dataset of specified parameters, suitable to generate datasets
        with linear correlation between a feature and a label. Only a 2 variable dataset
        can be created
        :param n: number of samples
        :param variance: variance you want in you data, higher the variance, weaker the correlation
        :param correlation: 'pos' or 'neg' (positive or negative), in hindsight, you are setting the
        sign of the slope of our model
        :param step: absolute difference between two data-points.
        :return:
        """
        val = 1
        y = []  # initialize empty label array
        for i in range(n):  # number of elements that will be appended to 'y'

            y_step = val + randrange(-variance, variance)  # element = 1 + random real number ∈ [-variance, variance)
            y.append(y_step)  # new element is appended to 'y'
            # Now depending on the correlation parameter, the next element base value is increased on decreased
            # depending upon the step parameter

            if correlation and correlation == 'pos':
                val += step
            elif correlation and correlation == 'neg':
                val -= step
            else:
                raise TypeError("Illegal correlation parameter or parameter not provided. Please use 'neg' or 'pos'")

        # Corresponding feature array is created
        X = [i for i in range(len(y))]

        return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

    def fit(self, X, y):
        """
        Fits linear model
        :param X: {array-like, sparse matrix} of shape (n_samples, 1)
                  Training data
        :param y: array-like of shape (n_samples,) or (n_samples, 1)
                  Target values. Will be cast to X's dtype if necessary
        :return: Intercept and slope of best fit line
        """
        self.slope = (np.mean(X * y) - (np.mean(X) * np.mean(y))) / (np.mean(X ** 2) - (np.mean(X) ** 2))
        self.intercept = np.mean(y) - (self.slope * np.mean(X))
        return self.intercept, self.slope

    def predict(self, X):
        """
        Predict using the linear model
        :param X: array_like or sparse matrix, shape (n_samples, 1)
                  Samples
        :return: An array of corresponding predicted values.
        """
        y_predicted = [((self.slope * x) + self.intercept) for x in X]
        return np.array(y_predicted, dtype=np.float64)

    class Quality:
        """
        This class is used to measure the accuracy and related parameters of the linear model
        created above.
        """
        @staticmethod
        def rmse(y, y_pred):
            """
            Root Mean Squared Error
            :param y: True values of the labels
            :param y_pred: Predicted values of the label
            :return: Root Mean Squared Error
            """
            return np.sqrt(np.mean((y_pred - y) ** 2))

        @staticmethod
        def mse(y, y_pred):
            """
            Mean Squared Error
            :param y: True values of the labels
            :param y_pred: Predicted values of the label
            :return: Mean Squared Error
            """
            return np.mean((y_pred - y) ** 2)

        @staticmethod
        def se(y, y_pred):
            """
            Sum of Squared Errors
            :param y: True values of the labels
            :param y_pred: Predicted values of the label
            :return: Sum of Squared Error
            """
            return sum((y_pred - y) ** 2)


# Usage of the class ->
if __name__ == '__main__':

    # Define a classifier and initialize the Quality subclass to measure the accuracy of you model.
    clf = SimpleLinearRegression()
    quality = SimpleLinearRegression.Quality()

    # Create training and testing datasets, you can also use sklearn.datasets.make_regression for this too
    # but I wanted to define a method myself.
    X_whole, y_whole = clf.create_dataset(50, 2, correlation='pos')
    X_train, y_train = X_whole[:len(X_whole)//2], y_whole[:len(y_whole)//2]
    X_test, y_test = X_whole[len(X_whole)//2:], y_whole[len(y_whole)//2:]

    # The fit() method returns intercept and slope in a tuple like this -> (intercept, slope)
    b, m = clf.fit(X_train, y_train)

    # Create the best fit line by making a list of all the features corresponding to estimated intercept and slope.
    best_fit_line = [((m * x) + b) for x in X_train]

    # Visualizing model over test data
    # A scatter plot and the aforementioned best-fit-line is plotted using matplotlib.pyplot
    plt.scatter(X_train, y_train)
    plt.plot(X_train, best_fit_line)
    plt.title("Best fit line for the training data")
    plt.show()

    # Making predictions over test data based on estimated intercept and slope
    print("Testing Accuracy on a new dataset... \n")
    y_predict = clf.predict(X_test)

    rmse = quality.rmse(y_test, y_predict)
    mse = quality.mse(y_test, y_predict)
    se = quality.se(y_test, y_predict)

    # Various error measures
    print("Root mean squared error = ", rmse)
    print("mean squared error = ", mse)
    print("squared error = ", se)
