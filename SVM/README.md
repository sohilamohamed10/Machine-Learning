
# SVM Implementation:

-Svm is a binary classifier and Iris data is a multi-class problem so we could use one vs all or one versus one approach to convert it to a binary classification problem . Since the data is balanced for the 3 class (each conatin 50 samples) one versus one approach  will more effective , For that reason one could choose ever to classes to train the classifier then the final decision (visualizing the accuarcy or error) can be taken as the majority vote of the three classifiers .
Since the targets are represented as strings I mapped one target to -1 and the other to 
1 to use it in the equation yi(w.xi+b).

-Used only the first 2 features to be able to visualize the data with the decision boundary.

-After that gradient descent concept is applied by assuming a number of 
iterations to optimize the weights and terminate when the cost function is almost 
constant .

Accuracy->70%