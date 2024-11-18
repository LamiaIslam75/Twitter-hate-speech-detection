import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np
from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1

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
    plt.savefig('smoothing_impact.png', bbox_inches='tight', dpi=300)
    plt.close()
    
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
        Dictionary containing evaluation metrics for both feature approaches
    """
    results = {}
    k = 1  # smoothing parameter
    
    # Test Feature Set 1: Binary Features
    print("\nTesting Feature Set 1: Binary Features (Word Presence)")
    class_priors, word_probs, vocab = features1(train_data, k)
    model1 = NaiveBayes(class_priors, word_probs, vocab)
    
    # Calculate F1 score for binary features
    f1_binary = f_1(model1, test_data)
    results['binary_features'] = {'f1': f1_binary}
    print(f"F1 Score (Binary Features): {f1_binary:.4f}")
    
    # Test Feature Set 2: Unigrams + Bigrams
    print("\nTesting Feature Set 2: Unigrams + Bigrams")
    class_priors, word_probs, vocab = features2(train_data, k)
    model2 = NaiveBayes(class_priors, word_probs, vocab)
    
    # Calculate F1 score for unigrams + bigrams
    f1_bigram = f_1(model2, test_data)
    results['unigram_bigram'] = {'f1': f1_bigram}
    print(f"F1 Score (Unigrams + Bigrams): {f1_bigram:.4f}")
    
    return results

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
    """
    Train and evaluate logistic regression model.
    
    Steps:
    1. Build word-to-index mapping from training data
    2. Convert data to feature matrices using featurize()
    3. Train logistic regression with specified parameters
    4. Evaluate on test set
    
    Args:
        train_data: List of (document, label) tuples for training
        test_data: List of (document, label) tuples for testing
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Step 1: Convert training data into features and build vocabulary
    X_train, Y_train = featurize(train_data)
    
    # Step 2: Convert test data into features using same vocabulary
    X_test, Y_test = featurize(test_data, train_data)
    
    # Step 3: Initialize and train model
    model = LogReg(eta=0.01, num_iter=10)  # 10 iterations as specified
    model.C = 0.1  # Set L2 regularization parameter
    model.train(X_train, Y_train)
    
    # Step 4: Evaluate model on test set
    test_predictions = model.predict(X_test)
    
    # Calculate accuracy
    correct = 0
    total = len(test_data)
    
    for i, (_, label) in enumerate(test_data):
        pred_label = 'offensive' if test_predictions[i] == 0 else 'nonoffensive'
        if pred_label == label:
            correct += 1
    
    accuracy = correct / total
    
    # Calculate F1 score
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    for i, (_, label) in enumerate(test_data):
        pred_label = 'offensive' if test_predictions[i] == 0 else 'nonoffensive'
        if label == 'offensive':
            if pred_label == 'offensive':
                true_pos += 1
            else:
                false_neg += 1
        else:
            if pred_label == 'offensive':
                false_pos += 1
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\nLogistic Regression Results:")
    print("===========================")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"Test F1 Score: {f1:.3f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'model': model
    }