import numpy									#for numpy arrays
from sklearn import tree						#for decision tree
from sklearn.naive_bayes import MultinomialNB	#for Naive Bayes
from sklearn.svm import LinearSVC				#for SVM
from sklearn.model_selection import KFold		#for K-fold cross validation
from sklearn.metrics import accuracy_score		#for classification accuracy
import matplotlib.pyplot as plt				#for plotting accuracies for all tests
import time 									#for computing running time

#GLOBAL VARIABLES
XList = []
YList = []
kf = 0
accuracyListDecisionTree = []
accuracyListNaiveBayes = []
accuracyListSVM = []


def parseText():
	#ADD GLOBAL VARIABLES
	global XList
	global YList
	global kf

	dataSet = open("connect-4.data", "r")
	
	for s in dataSet.readlines():
		s = s.split(',')
		sLast = s.pop()
		sLast = 1 if sLast=="win\n" else 0 if sLast=="draw\n" else -1

		XList.append( list(map((lambda x: 1 if x=='x' else 0 if x=='b' else 2), s)) )
		YList.append(sLast)

	XList = numpy.array(XList)
	YList = numpy.array(YList)

	kf = KFold(n_splits = 10, shuffle = False)	#Making sets from k fold cross validation
	kf.get_n_splits(XList)

def decisionTree():
	global XList
	global YList
	global kf
	global accuracyListDecisionTree

	startTime = time.time()

	output = open("decisionTreeOutput.txt", 'w')
	accuracyListDecisionTree = []

	count = 1
	for train_index, test_index in kf.split(XList):

		X_train, X_test = XList[train_index], XList[test_index]
		Y_train, Y_test = YList[train_index], YList[test_index]
		classifier = tree.DecisionTreeClassifier()										# decision tree classifier
		classifier = classifier.fit(X_train, Y_train)									# training tree classifier

		output.write("Training model " + str(count) + ": \n")
		output.write("\n\nTesting indices: \n")
		output.write(str(test_index))

		Y_pred = classifier.predict(X_test)
		Y_true = Y_test

		accuracyListDecisionTree.append(accuracy_score(Y_true, Y_pred))

		output.write("\n\nPredictions:\n")
		for j in range(len(Y_pred)):
			output.write(str(X_test[j]) + " : "+  str(Y_pred[j]) + '\n')

		output.write('\nAccuracy on this set: ' + str(accuracyListDecisionTree[-1]) + '\n\n')

		count += 1

	accuracyListDecisionTree = numpy.array(accuracyListDecisionTree)
	runningTime = time.time() - startTime

	output.write("\n\n Total time taken by Decision Tree: " + str(runningTime))
	output.write("\n Average accuracy: " + str(accuracyListDecisionTree.sum()/10))
	output.write("\n Highest accuracy: " + str(accuracyListDecisionTree.max()))
	output.close()

def naiveBayes():
	global XList
	global YList
	global kf
	global accuracyListNaiveBayes

	startTime = time.time()

	output = open("naiveBayesOutput.txt", 'w')
	accuracyListNaiveBayes = []

	count = 1
	for train_index, test_index in kf.split(XList):

		X_train, X_test = XList[train_index], XList[test_index]
		Y_train, Y_test = YList[train_index], YList[test_index]
		classifier = MultinomialNB(alpha = 0.5)									# naive bayes classifier
		classifier = classifier.fit(X_train, Y_train)									# training naive bayes classifier

		output.write("Training model " + str(count) + ": \n")	
		output.write("\n\nTesting indices: \n")
		output.write(str(test_index))

		Y_pred = classifier.predict(X_test)
		Y_true = Y_test

		accuracyListNaiveBayes.append(accuracy_score(Y_true, Y_pred))

		output.write("\n\nPredictions:\n")
		for j in range(len(Y_pred)):
			output.write(str(X_test[j]) + " : "+  str(Y_pred[j]) + '\n')

		output.write('\nAccuracy on this set: ' + str(accuracyListNaiveBayes[-1]) + '\n\n')

		count += 1

	accuracyListNaiveBayes = numpy.array(accuracyListNaiveBayes)
	runningTime = time.time() - startTime

	output.write("\n\n Total time taken by Naive Bayes classifier: " + str(runningTime))
	output.write("\n Average accuracy: " + str(accuracyListNaiveBayes.sum()/10))
	output.write("\n Highest accuracy: " + str(accuracyListNaiveBayes.max()))
	output.close()

def svm():
	global XList
	global YList
	global kf
	global accuracyListSVM

	startTime = time.time()

	output = open("svmOutput.txt", 'w')
	accuracyListSVM = []

	count = 1
	for train_index, test_index in kf.split(XList):

		X_train, X_test = XList[train_index], XList[test_index]
		Y_train, Y_test = YList[train_index], YList[test_index]
		classifier = LinearSVC()														# SVM classifier
		classifier = classifier.fit(X_train, Y_train)									# training SVM classifier

		output.write("Training model " + str(count) + ": \n")	
		output.write("\n\nTesting indices: \n")
		output.write(str(test_index))

		Y_pred = classifier.predict(X_test)
		Y_true = Y_test

		accuracyListSVM.append(accuracy_score(Y_true, Y_pred))

		output.write("\n\nPredictions:\n")
		for j in range(len(Y_pred)):
			output.write(str(X_test[j]) + " : "+  str(Y_pred[j]) + '\n')

		output.write('\nAccuracy on this set: ' + str(accuracyListSVM[-1]) + '\n\n')

		count += 1

	accuracyListSVM = numpy.array(accuracyListSVM)
	runningTime = time.time() - startTime

	output.write("\n\n Total time taken by SVM: " + str(runningTime))
	output.write("\n Average accuracy: " + str(accuracyListSVM.sum()/10))
	output.write("\n Highest accuracy: " + str(accuracyListSVM.max()))
	output.close()

def plot():
	global accuracyListDecisionTree
	global accuracyListNaiveBayes
	global accuracyListSVM

	plt.plot([1,2,3,4,5,6,7,8,9,10], accuracyListDecisionTree, color ='r', label ='Decision Tree Accuracy')
	plt.plot([1,2,3,4,5,6,7,8,9,10], accuracyListNaiveBayes, color ='g', label ='Naive Bayes Accuracy')
	plt.plot([1,2,3,4,5,6,7,8,9,10], accuracyListSVM, color ='b', label ='SVM Accuracy')

	plt.xlabel('Test Data number')
	plt.ylabel('Accuracy')
	plt.title("Comparison of different classifiers using 10-fold validation")
	plt.legend(loc='upper right')
	plt.savefig('figure-1.png')
	plt.show()

def main():
	parseText()
	decisionTree()
	naiveBayes()
	svm()
	plot()



if __name__ == '__main__':
  main() 