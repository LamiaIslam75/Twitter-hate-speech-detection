import matplotlib.pyplot as plt

from model.naivebayes import NaiveBayes, features1, features2
from model.logreg import LogReg, featurize
from evaluation import accuracy, f_1


def train_smooth(train_data, test_data):
    """
    Train classifier with different smoothing parameters and evaluate performance.
    
    Args:
        train_data: List of (document, label) tuples for training
        test_data: List of (document, label) tuples for testing
    """
    # Separate features and labels
    X_train, y_train = zip(*train_data)
    
    # Define range of k values to test (log scale)
    k_values = np.logspace(-3, 2, 20)  # From 0.001 to 100
    accuracies = []
    f1_scores = []
    
    # Test each k value
    for k in k_values:
        # Initialize and train model with current k
        clf = NaiveBayes(k=k)
        clf.fit(X_train, y_train)
        
        # Evaluate model
        acc = accuracy(clf, test_data)
        f1 = f_1(clf, test_data)
        
        accuracies.append(acc)
        f1_scores.append(f1)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(k_values, accuracies, 'b-', label='Accuracy')
    plt.semilogx(k_values, f1_scores, 'r-', label='F1 Score')
    plt.xlabel('Smoothing Parameter (k)')
    plt.ylabel('Score')
    plt.title('Impact of Smoothing Parameter on Model Performance')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig('smoothing_analysis.png')
    plt.close()
    
    # Print best results
    best_k_acc = k_values[np.argmax(accuracies)]
    best_k_f1 = k_values[np.argmax(f1_scores)]
    print(f"Best k for accuracy: {best_k_acc:.3f} (accuracy: {max(accuracies):.3f})")
    print(f"Best k for F1 score: {best_k_f1:.3f} (F1 score: {max(f1_scores):.3f})")



def train_feature_eng(train_data, test_data):
    # YOUR CODE HERE
    #     TODO:
    #         1) Improve on the basic bag of words model by changing
    #         the feature list of your model. Implement at least two
    #         variants using feature1 and feature2
    ########################### STUDENT SOLUTION ########################
    """
    Compare different feature engineering approaches.
    
    Args:
        train_data: List of (document, label) tuples for training
        test_data: List of (document, label) tuples for testing
    """
    # Separate documents and labels
    train_docs, train_labels = zip(*train_data)
    test_docs, test_labels = zip(*test_data)
    
    # Define feature engineering approaches
    feature_approaches = {
        'baseline': lambda x: x,  # Original features
        'stopwords_stem': features1,  # Stopwords removal + stemming
        'pos_bigrams': features2  # POS tags + bigrams
    }
    
    # Results storage
    results = {}
    
    # Test each feature engineering approach
    for name, feature_func in feature_approaches.items():
        # Process features
        X_train = [feature_func(doc) for doc in train_docs]
        X_test = [feature_func(doc) for doc in test_docs]
        
        # Train classifier
        clf = NaiveBayes(k=1.0)  # Using default k=1.0
        clf.fit(X_train, train_labels)
        
        # Evaluate
        test_data_processed = list(zip(X_test, test_labels))
        acc = accuracy(clf, test_data_processed)
        f1 = f_1(clf, test_data_processed)
        
        results[name] = {'accuracy': acc, 'f1': f1}
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Create grouped bar chart
    x = np.arange(len(feature_approaches))
    width = 0.35
    
    plt.bar(x - width/2, 
           [results[k]['accuracy'] for k in feature_approaches], 
           width, 
           label='Accuracy')
    plt.bar(x + width/2, 
           [results[k]['f1'] for k in feature_approaches], 
           width, 
           label='F1 Score')
    
    plt.xlabel('Feature Engineering Approach')
    plt.ylabel('Score')
    plt.title('Comparison of Feature Engineering Approaches')
    plt.xticks(x, feature_approaches.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('feature_engineering_comparison.png')
    plt.close()
    
    # Print results
    print("\nFeature Engineering Results:")
    print("============================")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 Score: {metrics['f1']:.3f}")
    
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