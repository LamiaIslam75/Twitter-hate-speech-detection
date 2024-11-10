def accuracy(classifier, data):
    """Computes the accuracy of a classifier on reference data.

    Args:
        classifier: A classifier with a predict() method
        data: Reference data containing (input, label) pairs

    Returns:
        The accuracy of the classifier on the test data, a float.
    """
    if not data:
        return 0.0
    
    correct = 0
    total = len(data)
    
    for item, true_label in data:
        predicted_label = classifier.predict(item)
        if predicted_label == true_label:
            correct += 1
            
    return correct / total



def f_1(classifier, data):
    """Computes the F_1-score of a classifier on reference data.

    Args:
        classifier: A classifier with a predict() method
        data: Reference data containing (input, label) pairs

    Returns:
        The F_1-score of the classifier on the test data, a float.
    """
    if not data:
        return 0.0
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for item, true_label in data:
        predicted_label = classifier.predict(item)
        
        if predicted_label == true_label == 1:
            true_positives += 1
        elif predicted_label == 1 and true_label == 0:
            false_positives += 1
        elif predicted_label == 0 and true_label == 1:
            false_negatives += 1
   
    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1