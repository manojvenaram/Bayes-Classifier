# Bayes-Classifier
## Aim:
To Construct a Bayes Classifier to classiy iris dataset using Python.
## Algorithm:
Input: 
- X: the training data, where each row represents a sample and each column represents a feature.
- y: the target labels for the training data.
- X_test: the testing data, where each row represents a sample and each column represents a feature.

Output:
- y_pred: the predicted labels for the testing data.

1. Create a BayesClassifier class with the following methods:
   a. __init__ method to initialize the Gaussian Naive Bayes classifier from scikit-learn.
   b. fit method to fit the classifier to the training data using the Gaussian Naive Bayes algorithm from scikit-learn.
   c. predict method to make predictions on the testing data using the fitted classifier from scikit-learn.
2. Load the Iris dataset using the load_iris function from scikit-learn.
3. Split the data into training and testing sets using the train_test_split function from scikit-learn.
4. Create a BayesClassifier instance.
5. Train the classifier on the training data using the fit method.
6. Make predictions on the testing data using the predict method.
7. Evaluate the classifier's accuracy using the accuracy_score function from scikit-learn.

## Program:
## Import the necessary libaries
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
```
## Create the class BayesClassfier
```
class BayesClassifier:
  def __init__(self) -> None:
    self.clf=GaussianNB()
  def fit(self,X,y):
    self.clf.fit(X,y)
  def predict(self,X):
    return self.clf.predict(X)
```
## Import the dataset
```
data=pd.read_csv("IRIS.csv")
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
```
## Split the data set 
```
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.33,random_state=38)
```
## Fit the Training data set
```
clf=BayesClassifier()
clf.fit(xtrain,ytrain)
```
## Predict with test data
```
ypred=clf.predict(xtest)
ypred
```
## Accuracy Score
```
accuracyscore=accuracy_score(ytest,ypred)
accuracyscore
```

## Output:
## Dataset
![](1.png)
## Accuracy Score
![](2.png)
## Result:
Hence, Bayes classifier for iris dataset is implemented successfully



