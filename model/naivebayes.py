class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    def __init__(self):
        """Initialises a new classifier."""
        pass
    ####################################################################


    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################
        # YOUR CODE HERE
        return None
        ############################################################


    @classmethod
    def train(cls, data, k=1):
        """Train a new classifier on training data using maximum
        likelihood estimation and additive smoothing.

        Args:
            cls: The Python class representing the classifier.
            data: Training data.
            k: The smoothing constant.

        Returns:
            A trained classifier, an instance of `cls`.
        """
        ##################### STUDENT SOLUTION #####################
        # YOUR CODE HERE
        return cls()
        ############################################################



def features1(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return None
    ##################################################################


def features2(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    ###################### STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    return None
    ##################################################################

