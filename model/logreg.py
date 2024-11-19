import numpy as np


class LogReg:
    def __init__(self, eta=0.01, num_iter=30, C=0.1):
        self.eta = eta
        self.num_iter = num_iter
        self.C = C

    def softmax(self, inputs):
        """
        Calculate the softmax for the given inputs (array)
        :param inputs: N x K matrix where N is number of instances and K is number of classes
        :return: N x K matrix of softmax probabilities
        """
        # Subtract max for numerical stability
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)


    def train(self, X, Y):
        """
        Train the logistic regression model using gradient descent
        :param X: N x F matrix of features
        :param Y: N x 2 matrix of one-hot encoded labels
        """
        # Initialize weights (F x 2 matrix, where F is number of features and 2 is number of classes)
        self.weights = np.zeros((X.shape[1], Y.shape[1]))
        
        n_samples = X.shape[0]
        batch_size = 100  # mini-batch size
        
        for epoch in range(self.num_iter):
            # Shuffle the data
            shuffle_idx = np.random.permutation(n_samples)
            X_shuffled = X[shuffle_idx]
            Y_shuffled = Y[shuffle_idx]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # Forward pass
                scores = np.dot(X_batch, self.weights)  # N x 2
                probs = self.softmax(scores)  # N x 2
                
                # Compute gradients
                diff = probs - Y_batch  # N x 2
                grad = np.dot(X_batch.T, diff)  # F x 2
                
                # Add L2 regularization gradient
                grad += self.C * self.weights
                
                # Update weights
                self.weights -= self.eta * grad


    def p(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in (log) probability prediction
        ################## STUDENT SOLUTION ########################
        """
        Calculate probability distributions for input instances
        :param X: N x F matrix of features
        :return: N x 2 matrix of probabilities
        """
        scores = np.dot(X, self.weights)
        return self.softmax(scores)
        ############################################################


    def predict(self, X):
        # YOUR CODE HERE
        #     TODO:
        #         1) Replace next line with prediction of best class
        ####################### STUDENT SOLUTION ####################
        """
        Predict classes for input instances
        :param X: N x F matrix of features
        :return: N-dimensional array of predicted classes (0 for offensive, 1 for nonoffensive)
        """
        probs = self.p(X)
        return np.argmax(probs, axis=1)
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
    return {word: idx for idx, word in enumerate(sorted(vocab))}
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
    # Build vocabulary from training data
    if train_data is None:
        train_data = data
    
    # Create vocabulary from training data
    vocab = set()
    for text, _ in train_data:
        vocab.update(text)
    
    # Build word to index mapping
    w2i = buildw2i(vocab)
    
    # Create feature matrix X
    X = np.zeros((len(data), len(w2i)))
    Y = np.zeros((len(data), 2))
    
    # Fill matrices
    for i, (text, label) in enumerate(data):
        # Fill X matrix - binary features for word presence
        for word in text:
            if word in w2i:
                X[i, w2i[word]] = 1
        
        # Fill Y matrix - one-hot encoding
        if label == 'offensive':
            Y[i] = [1, 0]
        else:  # nonoffensive
            Y[i] = [0, 1]
    
    return X, Y
    ##############################################################

