#-------------------------------------------------------------------------
# AUTHOR: Joshua Reyes
# FILENAME: svm.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

correct = 0
total = 0
highest_accuracy = 0
params = {}
#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

for value1 in c:
    for value2 in degree:
        for value3 in kernel:
           for value4 in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here
                svmCLF = svm.SVC(C = value1, degree = value2, kernel = value3, decision_function_shape = value4)
                #Fit SVM to the training data
                #--> add your Python code here
                svmCLF.fit(X_training, y_training)
                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                #--> add your Python code here
                
                for(X_testSample, y_testSample) in zip(X_test, y_test):
                    svmCLF.predict([X_testSample])
                    if  svmCLF.predict([X_testSample]) == y_testSample:
                        correct += 1
                    total += 1
                accuracy = correct/total
                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    params = {"C": value1, "degree": value2, "kernel": value3, "decision_function_shape":value4}
                    print(f"Highest SVM accuracy so far: {highest_accuracy}, Parameters: C={params['C']}, degree={params['degree']}, kernel={params['kernel']}, decision_function_shape={params['decision_function_shape']}")

                correct = 0
                total = 0






