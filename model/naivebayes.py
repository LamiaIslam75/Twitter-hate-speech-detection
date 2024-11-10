import math
from collections import defaultdict, Counter

class NaiveBayes(object):

    ######################### STUDENT SOLUTION #########################
    # YOUR CODE HERE
    def __init__(self, class_priors=None, word_probs=None, vocabulary=None):
        """Initialises a new classifier.
        
        Args:
            class_priors: Dictionary of class prior probabilities
            word_probs: Dictionary of word probabilities per class
            vocabulary: Set of words from training data
        """
        self.class_priors = class_priors or {}
        self.word_probs = word_probs or {}
        self.vocabulary = vocabulary or set()
    ####################################################################


    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
        ################## STUDENT SOLUTION ########################
        scores = {}
        
        # Calculate score for each class
        for class_label in self.class_priors:
            # Start with log prior probability
            score = math.log(self.class_priors[class_label])
            
            # Add log likelihood for each word
            for word in x:
                if word in self.vocabulary:
                    score += math.log(self.word_probs[class_label].get(word, 1e-10))
            
            scores[class_label] = score
        
        # Return class with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
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
        # Initialize counters
        class_counts = Counter()
        word_counts = defaultdict(Counter)
        vocabulary = set()

        # Count occurrences
        total_documents = len(data)
        for text, label in data:
            class_counts[label] += 1
            for word in text:
                word_counts[label][word] += 1
                vocabulary.add(word)

        # Calculate class priors
        class_priors = {label: count/total_documents 
                       for label, count in class_counts.items()}

        # Calculate word probabilities with smoothing
        word_probs = {}
        vocab_size = len(vocabulary)
        
        for label in class_counts:
            word_probs[label] = {}
            total_words = sum(word_counts[label].values())
            
            for word in vocabulary:
                numerator = word_counts[label][word] + k
                denominator = total_words + k * vocab_size
                word_probs[label][word] = numerator / denominator

        return cls(class_priors, word_probs, vocabulary)
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
    # Convert data to binary features (word presence)
    binary_data = [(list(set(text)), label) for text, label in data]
    
    # Train classifier with binary features
    classifier = NaiveBayes.train(binary_data, k)
    
    return classifier.class_priors, classifier.word_probs, classifier.vocabulary
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
    # Create data with unigrams and bigrams
    processed_data = []
    for text, label in data:
        # Add unigrams
        features = text.copy()
        
        # Add bigrams
        for i in range(len(text) - 1):
            bigram = f"{text[i]}_{text[i+1]}"
            features.append(bigram)
            
        processed_data.append((features, label))
    
    # Train classifier with enhanced features
    classifier = NaiveBayes.train(processed_data, k)
    
    return classifier.class_priors, classifier.word_probs, classifier.vocabulary
    ##################################################################

