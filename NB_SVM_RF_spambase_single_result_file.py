"""
Created on Sun Apr 19 15:11:48 2020
@author: Pramod
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
#from sklearn.metrics import accuracy_score
# Importing the datasets
#datasets = pd.read_csv('Social_Network_Ads.csv')
datasets = pd.read_csv('spambase.csv')
X = datasets.iloc[:, 0:56].values
Y = datasets.iloc[:, 57].values
#print(X)
#print(Y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)
#GaussianNB
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_Train, Y_Train)
Y_Pred = classifier.predict(X_Test)
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report,accuracy_score, f1_score
rst_file = open("NB_SVM_RF_Result_spambase.txt","a")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write("\n"+str(datetime.datetime.now())+"\n")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write('\nClassification Result of GaussianNB')
rst_file.write('\nAccuracy :'+str(accuracy_score(Y_Test, Y_Pred)))
rst_file.write('\nF1 score :'+str(f1_score(Y_Test, Y_Pred)))
rst_file.write('\nRecall :'+str(recall_score(Y_Test, Y_Pred)))
rst_file.write('\nPrecision :'+str(precision_score(Y_Test, Y_Pred)))
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write('\nclasification report : \n'+str(classification_report(Y_Test,Y_Pred)))
#rst_file.write("\nAccuracy score of train : "+str(accuracy_score(Y_Train, classifier.predict(X_Train))*100))
#rst_file.write("\nAccuracy score of test : "+str(accuracy_score(Y_Test, Y_Pred)*100))
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write("\nConfusion Matrix : \n"+str(confusion_matrix(Y_Test, Y_Pred))+"\n")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.close()

# Fitting the classifier into the Training set

#SVM Linear

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_Train, Y_Train)
# Predicting the test set results
Y_Pred = classifier.predict(X_Test)
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report,accuracy_score, f1_score
rst_file = open("NB_SVM_RF_Result_spambase.txt","a")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write("\n"+str(datetime.datetime.now())+"\n")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write('\nClassification Result of Linear SVM')
rst_file.write('\nAccuracy :'+str(accuracy_score(Y_Test, Y_Pred)))
rst_file.write('\nF1 score :'+str(f1_score(Y_Test, Y_Pred)))
rst_file.write('\nRecall :'+str(recall_score(Y_Test, Y_Pred)))
rst_file.write('\nPrecision :'+str(precision_score(Y_Test, Y_Pred)))
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write('\nclasification report : \n'+str(classification_report(Y_Test,Y_Pred)))
#rst_file.write("\nAccuracy score of train : "+str(accuracy_score(Y_Train, classifier.predict(X_Train))*100))
#rst_file.write("\nAccuracy score of test : "+str(accuracy_score(Y_Test, Y_Pred)*100))
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write("\nConfusion Matrix : \n"+str(confusion_matrix(Y_Test, Y_Pred))+"\n")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.close()
#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_Train,Y_Train)
# Predicting the test set results
Y_Pred = classifier.predict(X_Test)
from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report,accuracy_score, f1_score
rst_file = open("NB_SVM_RF_Result_spambase.txt","a")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write("\n"+str(datetime.datetime.now())+"\n")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write('\nClassification Result of Random Forest')
rst_file.write('\nAccuracy :'+str(accuracy_score(Y_Test, Y_Pred)))
rst_file.write('\nF1 score :'+str(f1_score(Y_Test, Y_Pred)))
rst_file.write('\nRecall :'+str(recall_score(Y_Test, Y_Pred)))
rst_file.write('\nPrecision :'+str(precision_score(Y_Test, Y_Pred)))
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write('\nclasification report : \n'+str(classification_report(Y_Test,Y_Pred)))
#rst_file.write("\nAccuracy score of train : "+str(accuracy_score(Y_Train, classifier.predict(X_Train))*100))
#rst_file.write("\nAccuracy score of test : "+str(accuracy_score(Y_Test, Y_Pred)*100))
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.write("\nConfusion Matrix : \n"+str(confusion_matrix(Y_Test, Y_Pred))+"\n")
rst_file.write('\n-------------------------------------------------------------------------')
rst_file.close()