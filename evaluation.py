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
        The macro-averaged F1-score of the classifier on the test data, a float.
    """
    if not data:
        return 0.0
    
    # Get all unique classes from the data
    labels = set(label for _, label in data)
    
    # Calculate F1 score for each class
    f1_scores = []
    for label in labels:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for item, true_label in data:
            predicted_label = classifier.predict(item)
            
            if predicted_label == label and true_label == label:
                true_positives += 1
            elif predicted_label == label and true_label != label:
                false_positives += 1
            elif predicted_label != label and true_label == label:
                false_negatives += 1
        
        # Calculate precision and recall for this class
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Calculate F1 score for this class
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    
    # Return macro-averaged F1 score
    return sum(f1_scores) / len(f1_scores)