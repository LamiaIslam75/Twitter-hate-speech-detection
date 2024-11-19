# Impact of Smoothing Parameter on Naive Bayes Model

![Smoothing Parameter Impact](smoothing_impact.png)

When k=1 (best performance):
- Provides optimal balance between seen and unseen words
- Gives reasonable probability to unseen words without overly discounting seen words
- Accuracy (0.8572) and F1 (0.7160) are highest because the model can handle both common and rare words effectively

When k is small (k < 1):
- Model relies too heavily on observed frequencies
- Gives very low probabilities to unseen words
- Leads to lower accuracy and F1 scores because the model struggles with new or rare words in the test set

When k is large (k > 1):
- Over-smooths the probabilities
- Makes word probabilities too uniform across classes
- Sharp drop in F1 score (down to 0.4519 at k=100) indicates model loses ability to distinguish between offensive and non-offensive text
- Large k essentially erases the meaningful differences between word frequencies in different classes

As a result, k=1 works best because it provides just enough smoothing to handle unseen words while preserving the discriminative power of the observed word frequencies in the training data.