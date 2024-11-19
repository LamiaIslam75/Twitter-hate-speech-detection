import math
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag
import nltk

# Download required NLTK data
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger_eng')

class NaiveBayes(object):
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
    
    def predict(self, x):
        """Predicts the class for a document.

        Args:
            x: A document, represented as a list of words.

        Returns:
            The predicted class, represented as a string.
        """
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

def features4(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
    # Convert data to binary features (word presence)
    binary_data = [(list(set(text)), label) for text, label in data]
    
    # Train classifier with binary features
    classifier = NaiveBayes.train(binary_data, k)
    
    return classifier.class_priors, classifier.word_probs, classifier.vocabulary


def features3(data, k=1):
    """
    Your feature of choice for Naive Bayes classifier.

    Args:
        data: Training data.
        k: The smoothing constant.

    Returns:
        Parameters for Naive Bayes classifier, which can
        then be used to initialize `NaiveBayes()` class
    """
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




def features1(data, k=1):
    """
    Feature engineering using minimal preprocessing and binary features.
    Keeps important stop words and uses word presence rather than frequency.
    """
    # List of stop words to remove (keep potentially important ones)
    stop_words = set(stopwords.words('english'))
    important_words = {
        # Negation words
        'not', 'no', 'never', 'none', 'nobody', 'nothing',
        
        # Intensifiers
        'very', 'so', 'too', 'more', 'most'
    }
    stop_words = stop_words - important_words
    
    processed_data = []
    for text, label in data:
        # Keep original case for offensive terms
        # Only remove very common stop words
        processed_text = [
            word for word in text 
            if word.lower() not in stop_words or word.lower() in important_words
        ]
        # Convert to binary features (presence/absence)
        binary_features = list(set(processed_text))
        processed_data.append((binary_features, label))
    
    classifier = NaiveBayes.train(processed_data, k)
    return classifier.class_priors, classifier.word_probs, classifier.vocabulary

def features2(data, k=1):
    """
    Feature engineering focusing on offensive language patterns.
    Creates bigrams when:
    1. An intensifier is followed by any word
    2. A negation word is followed by any word
    3. Any word followed by offensive-indicating words
    """
    # Words that often modify the intensity or sentiment
    intensifiers = {
        'so', 'really', 'very', 'too', 'more', 'most',
        'absolutely', 'totally', 'completely'
    }
    
    # Negation words that might modify meaning
    negations = {
        'not', 'no', 'never', 'none', 'nothing', 
        'nobody', "don't", "doesn't", "didn't"
    }
    
    processed_data = []
    
    for text, label in data:
        features = []
        # Keep original words
        features.extend(text)
        
        # Add bigrams for specific patterns
        for i in range(len(text) - 1):
            word = text[i].lower()
            next_word = text[i+1].lower()
            
            # Create bigrams if:
            # 1. First word is an intensifier
            # 2. First word is a negation
            if (word in intensifiers or word in negations):
                bigram = f"{text[i]}_{text[i+1]}"
                features.append(bigram)
        
        processed_data.append((features, label))
    
    classifier = NaiveBayes.train(processed_data, k)
    return classifier.class_priors, classifier.word_probs, classifier.vocabulary