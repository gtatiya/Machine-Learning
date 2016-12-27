# k Nearest Neighbors
- Evaluating Decision Trees (J48) using Weka
- Reading Data Files
- Implementing k Nearest Neighbors (kNN) Algorithm
- Evaluating kNN with respect to k
- Feature Selection for kNN

In this project, I have Implemented the k nearest neighbors algorithm (kNN) and compared its accuracy with the decision tree learning algorithm, and evaluate kNN’s sensitivity to the value of k and the relevance/number of features. I have used the Weka system for getting decision trees (J48) accuracies.
<br>
Generally, the accuracy of kNN initially increase with the number of k then decreases for large values of k. For all datasets, except spambase decision trees (J48) performs better than kNN. In general, the performance of kNN increases with n, the number of features selected. For some values of n, kNN performs better than decision trees (J48), but mostly decision trees (J48) outperforms kNN.