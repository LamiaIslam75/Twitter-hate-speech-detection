import numpy as np


class LogReg:
    def __init__(self, eta=0.01, num_iter=30):
        self.eta = eta
        self.num_iter = num_iter

    def softmax(self, inputs):
        """
        Calculate the softmax for the give inputs (array)
        :param inputs:
        :return:
        """
        # TODO: adapt for your solution
        return np.exp(inputs) / float(sum(np.exp(inputs)))


    def train(self, X, Y):

        #################### STUDENT SOLUTION ###################

        # weights initialization
        self.weights = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            # YOUR CODE HERE
            #     TODO:
            #         1) Fill in iterative updating of weights
            pass
        return None
        #########################################################


    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        return 0.0
        ############################################################


    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        return None
        #############################################################


def buildw2i(vocab):
    """
    Create indexes for 'featurize()' function.

    Args:
        vocab: vocabulary constructed from the training set.

    Returns:
        Dictionaries with word as the key and index as its value.
    """
    # YOUR CODE HERE
    #################### STUDENT SOLUTION ######################
    return None
    ############################################################


def featurize(data, train_data=None):
    """
    Convert data into X and Y where X is the input and
    Y is the label.

    Args:
        data: Training or test data.
        train_data: Reference data to build vocabulary from.

    Returns:
        Matrix X and Y.
    """
    # YOUR CODE HERE
    ##################### STUDENT SOLUTION #######################
    return None
    ##############################################################

