import numpy as np
from collections import defaultdict


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def build_w2i(documents):
    """Build word to index mapping from documents."""
    w2i = defaultdict(lambda: len(w2i))
    for doc in documents:
        for word in doc:
            w2i[word]  # Just access to add to dictionary if not present
    return dict(w2i)  # Convert to regular dict to freeze vocabulary


def featurize(documents, labels, w2i):
    """
    Convert documents and labels into matrix format.
    
    Args:
        documents: List of documents (each document is a list of words)
        labels: List of labels ('offensive' or 'nonoffensive')
        w2i: Word to index mapping
        
    Returns:
        X: N x F feature matrix
        Y: N x 2 one-hot encoded label matrix
    """
    N = len(documents)
    F = len(w2i)
    
    # Initialize feature matrix
    X = np.zeros((N, F))
    
    # Fill feature matrix
    for i, doc in enumerate(documents):
        for word in doc:
            if word in w2i:
                X[i, w2i[word]] = 1
    
    # Create one-hot encoded label matrix
    Y = np.zeros((N, 2))
    for i, label in enumerate(labels):
        Y[i, 0] = 1 if label == 'offensive' else 0
        Y[i, 1] = 1 if label == 'nonoffensive' else 0
    
    return X, Y


class LogReg:
    def __init__(self, num_features, learning_rate=0.01, C=0.1):
        """
        Initialize logistic regression classifier.
        
        Args:
            num_features: Number of features
            learning_rate: Learning rate for gradient descent
            C: L2 regularization parameter
        """
        self.W = np.zeros((num_features, 2))  # Weight matrix
        self.b = np.zeros(2)  # Bias terms
        self.lr = learning_rate
        self.C = C
    
    def p(self, X):
        """
        Compute class probabilities for input matrix X.
        
        Args:
            X: N x F feature matrix
            
        Returns:
            N x 2 probability matrix
        """
        scores = np.dot(X, self.W) + self.b
        return softmax(scores)
    
    def predict(self, X):
        """
        Predict classes for input matrix X.
        
        Args:
            X: N x F feature matrix
            
        Returns:
            List of predicted labels
        """
        probs = self.p(X)
        predictions = np.argmax(probs, axis=1)
        return ['offensive' if p == 0 else 'nonoffensive' for p in predictions]
    
    def train(self, X, Y, max_iter=10, batch_size=100):
        """
        Train the model using mini-batch gradient descent.
        
        Args:
            X: N x F feature matrix
            Y: N x 2 one-hot encoded label matrix
            max_iter: Number of epochs
            batch_size: Size of mini-batches
        """
        N = X.shape[0]
        
        for epoch in range(max_iter):
            # Shuffle data
            shuffle_idx = np.random.permutation(N)
            X_shuffled = X[shuffle_idx]
            Y_shuffled = Y[shuffle_idx]
            
            # Mini-batch training
            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # Forward pass
                probs = self.p(X_batch)
                
                # Compute gradients
                grad_W = np.dot(X_batch.T, (probs - Y_batch)) / batch_size
                grad_b = np.mean(probs - Y_batch, axis=0)
                
                # Add L2 regularization
                grad_W += self.C * self.W
                
                # Update parameters
                self.W -= self.lr * grad_W
                self.b -= self.lr * grad_b
