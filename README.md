## DMML ASSIGNMENT 2
Submission: 6th October, 2019

Ekanshdeep Gupta, BMC201710  
Samarth Ramesh, BMC201722

## ABOUT THE PROGRAM

We created a Python3 program to create three classifiers using Decision Trees, Naive Bayes Algorithm and Support Vector Machines for the Connect-4 dataset: http://archive.ics.uci.edu/ml/datasets/Connect-4

We read as input the set of all legal positions in the game of connect-4 in which neither player has won yet, and in which the next move is not forced. The outcome class is the theoretical value of the game from the perspective of the first player.

To evaluate the classifiers, we used 10-fold cross validation on the given data. We then compared the accuracies of the different classifiers for each of the 10 splits.

## PARAMETERS

For the different datasets, we used different frequencies:

- For Kos, we used the frequencies 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1
- For Enron, we used the frequencies 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.04, 0.03
- For NIPS, we used the frequencies 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45

For each frequency, the number of frequent itemsets, running time and the itemsets are all stored in the corresponding output files.

## LIBRARIES

- numpy: for numpy arrays
- sklearn: for decision tree
- sklearn.naive_bayes: for Naive Bayes
- sklearn.svm: for SVM
- sklearn.model_selection: for K-fold cross validation
- sklearn.metrics: for classification accuracy
- matplotlib.pyplot: for plotting accuracies for all tests
- time: for computing running time

## ALGORITHM

The program accepts the connect-4 training data and parses it into numpy arrays. It then uses KFold function to make 10 training/test splits of the data. The three clasifiers are then built using the DecisionTreeClassifier, MultinomialNB and LinearSVC functions and are trained by the fit function on each split. We then use the predict function to compare the performance of the model on the test data. These acccuracies are then compared and graphed to get a comprehensive overview about the performance of the different models.

## ACCURACY AND RUNNING TIME

Following is the accuracy graph obtained by 10-fold cross validation of all the three classifiers:

<div align="center" markdown="1">
<img src="figure-1.png" alt="Comparison of different classifiers using 10-fold validation"  style="height:90%; width:90%">
</div>

<table>
  <thead>
    <tr>
      <td rowspan=2>Classifier</td>
      <td rowspan=2>Accuracy</td>
      <td colspan=2>Time reqired</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Decision Tree</td>
      <td>0.6807</td>
      <td>31.7 sec</td>
    </tr>
    <tr>
      <td>Naive Bayes</td>
      <td>0.6131</td>
      <td>27.28 sec</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>0.6551</td>
      <td>331.37 sec</td>
    </tr>
	</tbody>
</table>