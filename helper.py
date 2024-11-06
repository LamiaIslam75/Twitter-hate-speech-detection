import matplotlib.pyplot as plt

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1


def train_smooth(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Re-train Naive Bayes while varying smoothing parameter k,
    #         then evaluate on test_data.
    #         2) Plot a graph of the accuracy and/or f-score given
    #         different values of k and save it, don't forget to include
    #         the graph for your submission.

    ######################### STUDENT SOLUTION #########################
    pass
    ####################################################################



def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################
    pass
    #####################################################################



def train_logreg(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) First, assign each word in the training set a unique integer index
    #         with `buildw2i()` function (in model/logreg.py, not here)
    #         2) Now that we have `buildw2i`, we want to convert the data into
    #         matrix where the element of the matrix is 1 if the corresponding
    #         word appears in a document, 0 otherwise with `featurize()` function.
    #         3) Train Logistic Regression model with the feature matrix for 10
    #         iterations with default learning rate eta and L2 regularization
    #         with parameter C=0.1.
    #         4) Evaluate the model on the test set.
    ########################### STUDENT SOLUTION ########################
    pass
    #####################################################################
