# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:41:58 2024

@author: hp
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def train_evaluate_models(features, target, cv, test_size, random_state):
    acc = []
    model = []

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=test_size, random_state=random_state)

    # Decision Tree
    DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=random_state, max_depth=5)
    DecisionTree.fit(Xtrain, Ytrain)
    predicted_values = DecisionTree.predict(Xtest)
    acc.append(accuracy_score(Ytest, predicted_values))
    model.append('Decision Tree')

    # Naive Bayes
    NaiveBayes = GaussianNB()
    NaiveBayes.fit(Xtrain, Ytrain)
    predicted_values = NaiveBayes.predict(Xtest)
    acc.append(accuracy_score(Ytest, predicted_values))
    model.append('Naive Bayes')

    # SVM
    SVM = SVC(gamma='auto')
    SVM.fit(Xtrain, Ytrain)
    predicted_values = SVM.predict(Xtest)
    acc.append(accuracy_score(Ytest, predicted_values))
    model.append('SVM')

    # Logistic Regression
    LogReg = LogisticRegression(random_state=random_state)
    LogReg.fit(Xtrain, Ytrain)
    predicted_values = LogReg.predict(Xtest)
    acc.append(accuracy_score(Ytest, predicted_values))
    model.append('Logistic Regression')

    return acc, model

# Load your dataset
PATH = "C:/Users/hp/Downloads/Crop_recommendation.csv"
df = pd.read_csv(PATH)

# Define features and target
features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Set up comparison table
comparison_table = pd.DataFrame(columns=['CV', 'Test Size', 'Random State', 'Model', 'Accuracy'])

# Vary parameters
cv_values = [3, 5]
test_sizes = [0.2, 0.3]
random_states = [1, 2]

iteration = 1
print("********************************************************************")
for cv in cv_values:
    for test_size in test_sizes:
        for random_state in random_states:
            acc, model = train_evaluate_models(features, target, cv, test_size, random_state)
            accuracy_models = dict(zip(model, acc))
            for k, v in accuracy_models.items():
                comparison_table = comparison_table.append({'CV': cv, 'Test Size': test_size, 'Random State': random_state, 'Model': k, 'Accuracy': v}, ignore_index=True)
            iteration += 1
            if iteration > 2:
                break
        if iteration > 5:
            break
    if iteration > 5:
        break

print(comparison_table)
print("********************************************************************")
