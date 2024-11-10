import numpy as np
from collections import defaultdict, Counter
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class NaiveBayes:
    def __init__(self, k=1.0):
        """Initialize Naive Bayes classifier with smoothing parameter k.
        
        Args:
            k (float): Smoothing parameter for additive (Laplace) smoothing
        """
        self.k = k
        self.vocab = set()  # Vocabulary from training data
        self.class_probs = {}  # P(class)
        self.word_probs = {}  # P(word|class)
        self.classes = set()  # Set of all possible classes
        
    def fit(self, X, y):
        """Train the classifier on the training data.
        
        Args:
            X: List of documents (each document is a list of words)
            y: List of corresponding labels
        """
        # Reset internal states
        self.vocab = set()
        self.class_probs = {}
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.classes = set(y)
        
        # Build vocabulary from training data
        for doc in X:
            self.vocab.update(doc)
            
        # Calculate class probabilities P(class)
        total_docs = len(y)
        class_counts = Counter(y)
        for c in self.classes:
            self.class_probs[c] = math.log(class_counts[c] / total_docs)
            
        # Calculate word probabilities P(word|class) with smoothing
        word_counts = {c: defaultdict(int) for c in self.classes}
        total_words = {c: 0 for c in self.classes}
        
        # Count words in each class
        for doc, label in zip(X, y):
            for word in doc:
                word_counts[label][word] += 1
                total_words[label] += 1
        
        # Calculate smoothed log probabilities
        vocab_size = len(self.vocab)
        for c in self.classes:
            denominator = total_words[c] + self.k * vocab_size
            for word in self.vocab:
                numerator = word_counts[c][word] + self.k
                self.word_probs[c][word] = math.log(numerator / denominator)
                
    def predict(self, document):
        """Predict the class for a single document.
        
        Args:
            document: List of words
            
        Returns:
            Predicted class label
        """
        scores = {c: self.class_probs[c] for c in self.classes}
        
        # Only consider known words (ignore unknown words)
        known_words = [word for word in document if word in self.vocab]
        
        # Calculate score for each class
        for c in self.classes:
            for word in known_words:
                scores[c] += self.word_probs[c][word]
                
        # Return class with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def predict_batch(self, documents):
        """Predict classes for multiple documents.
        
        Args:
            documents: List of documents (each document is a list of words)
            
        Returns:
            List of predicted class labels
        """
        return [self.predict(doc) for doc in documents]
    

def features1(document):
    """
    Feature engineering variant 1: Remove stopwords and apply stemming
    
    Args:
        document: Input text document
        
    Returns:
        List of processed features
    """
    # Initialize stemmer and get stopwords
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Tokenize if the input is a string
    if isinstance(document, str):
        words = word_tokenize(document.lower())
    else:
        words = document
    
    # Remove stopwords and apply stemming
    features = [stemmer.stem(word) for word in words 
               if word.lower() not in stop_words 
               and word.isalnum()]
    
    return features

def features2(document):
    """
    Feature engineering variant 2: POS tags and bigrams
    
    Args:
        document: Input text document
        
    Returns:
        List of processed features
    """
    # Tokenize if the input is a string
    if isinstance(document, str):
        words = word_tokenize(document.lower())
    else:
        words = document
    
    # Get POS tags
    pos_tags = pos_tag(words)
    
    # Create features
    features = []
    
    # Add POS-tagged words
    features.extend([f"{word}_{tag}" for word, tag in pos_tags])
    
    # Add bigrams
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    features.extend(bigrams)
    
    return features