import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np
from model.naivebayes import NaiveBayes, features1, features2, features3
from model.logreg import LogReg, featurize, buildw2i
from evaluation import accuracy, f_1
from pathlib import Path

def train_smooth(train_data, test_data):
    """
    Train Naive Bayes with different smoothing parameters and evaluate performance.
    
    Args:
        train_data: Training dataset
        test_data: Test dataset
    """
    # Define range of k values to test (log scale)
    k_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # Store results
    accuracies = []
    f1_scores = []
    
    # Train and evaluate for each k value
    for k in k_values:
        # Train classifier with current k
        classifier = NaiveBayes.train(train_data, k=k)
        
        # Evaluate performance
        acc = accuracy(classifier, test_data)
        f1 = f_1(classifier, test_data)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"k={k:.4f}: Accuracy={acc:.4f}, F1={f1:.4f}")

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy and F1 scores
    ax.semilogx(k_values, accuracies, 'b-o', label='Accuracy')
    ax.semilogx(k_values, f1_scores, 'r-o', label='F1 Score')
    
    # Customize plot
    ax.set_xlabel('Smoothing Parameter (k)')
    ax.set_ylabel('Score')
    ax.set_title('Impact of Smoothing Parameter on Model Performance')
    ax.grid(True)
    ax.legend()
    
    # Add value labels
    for i, k in enumerate(k_values):
        ax.annotate(f'{accuracies[i]:.3f}', 
                   (k, accuracies[i]), 
                   textcoords="offset points", 
                   xytext=(0,10), 
                   ha='center')
        ax.annotate(f'{f1_scores[i]:.3f}', 
                   (k, f1_scores[i]), 
                   textcoords="offset points", 
                   xytext=(0,-15), 
                   ha='center')
    
    # Save plot
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save plot in the output directory
    output_path = output_dir / 'smoothing_impact.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    
    # Return best accuracy and corresponding k value
    best_acc = max(accuracies)
    best_k = k_values[accuracies.index(best_acc)]
    
    print(f"\nBest accuracy {best_acc:.4f} achieved with k={best_k}")
    
    
    return best_acc, best_k

def train_feature_eng(train_data, test_data):
    """
    Train and evaluate Naive Bayes models with different feature engineering approaches.
    
    Args:
        train_data: List of (text, label) tuples for training
        test_data: List of (text, label) tuples for testing
    
    Returns:
        Dictionary containing evaluation metrics for feature approaches
    """
    results = {}
    k = 1  # smoothing parameter
    
    # Test Feature Set 1: Stop Words Removal
    print("\nTesting Feature Set 1: Stop Words Removal")
    class_priors, word_probs, vocab = features1(train_data, k)
    model1 = NaiveBayes(class_priors, word_probs, vocab)
    
    acc_stem_stop = accuracy(model1, test_data)
    f1_stem_stop = f_1(model1, test_data)
    results['stopwords'] = {
        'accuracy': acc_stem_stop,
        'f1': f1_stem_stop,
        'description': 'Stop Words Removal'
    }
    print(f"Accuracy: {acc_stem_stop:.3f}")
    print(f"F_1: {f1_stem_stop:.3f}")
    
    # Test Feature Set 2: Bigrams
    print("\nTesting Feature Set 2: Bigrams")
    class_priors, word_probs, vocab = features2(train_data, k)
    model2 = NaiveBayes(class_priors, word_probs, vocab)
    
    acc_pos_bigram = accuracy(model2, test_data)
    f1_pos_bigram = f_1(model2, test_data)
    results['bigrams'] = {
        'accuracy': acc_pos_bigram,
        'f1': f1_pos_bigram,
        'description': 'Bigrams'
    }
    print(f"Accuracy: {acc_pos_bigram:.3f}")
    print(f"F_1: {f1_pos_bigram:.3f}")

     # Test Feature Set 2: Bigrams
    print("\nTesting Feature Set 3: Stemming")
    class_priors, word_probs, vocab = features3(train_data, k)
    model3 = NaiveBayes(class_priors, word_probs, vocab)
    
    acc_stem = accuracy(model3, test_data)
    f1_stem = f_1(model3, test_data)
    results['bigrams'] = {
        'accuracy': acc_stem,
        'f1': f1_stem,
        'description': 'Stemming'
    }
    print(f"Accuracy: {acc_stem:.3f}")
    print(f"F_1: {f1_stem:.3f}")
    
    return results


def train_logreg(train_data, test_data):
    """
    Train and evaluate a Logistic Regression model on the provided data.
    
    Args:
        train_data: List of (text, label) tuples for training
        test_data: List of (text, label) tuples for testing
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # 1. Convert training data to feature matrices
    X_train, Y_train = featurize(train_data)
    
    # 2. Initialize and train logistic regression model
    model = LogReg(eta=0.01, num_iter=10, C=0.1)
    model.train(X_train, Y_train)
    
    # 3. Create a wrapper class to match the classifier interface expected by
    # the evaluation functions
    class LogRegWrapper:
        def __init__(self, logreg_model, w2i):
            self.model = logreg_model
            self.w2i = w2i
            
        def predict(self, text):
            # Convert text to feature vector
            x = np.zeros(len(self.w2i))
            for word in text:
                if word in self.w2i:
                    x[self.w2i[word]] = 1
            
            # Reshape to 2D array as expected by the model
            x = x.reshape(1, -1)
            
            # Get prediction
            pred = self.model.predict(x)[0]
            return 'offensive' if pred == 0 else 'nonoffensive'
    
    # 4. Create wrapper instance with vocabulary from training data
    vocab = list(set(word for text, _ in train_data for word in text))
    w2i = buildw2i(vocab)
    wrapped_model = LogRegWrapper(model, w2i)
    
    # 5. Evaluate model
    acc = accuracy(wrapped_model, test_data)
    f1 = f_1(wrapped_model, test_data)
    
    # 6. Print and return results
    print(f"\nLogistic Regression Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': acc,
        'f1': f1
    }