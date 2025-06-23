#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 12:55:042 2025

@author: juanitagomez
"""
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler # needed when uncommenting for UCSD
import numpy as np
np.random.seed(42)

# Method 1: Least Square Regression ------------------------------------------


def least_squares(A, M, all_data, acronym, method="embeddings"):
    
    
    # Objective function to minimize the squared error
    def objective(w):
        predictions = A @ w  # Calculate the scores for each repository
        return np.sum((M - predictions) ** 2)  # Least squares error
    
    
    # # Constraints: weights must be between 0 and 1
    constraints = [
        {'type': 'ineq', 'fun': lambda w: w},        # w >= 0
        {'type': 'ineq', 'fun': lambda w: 1 - w}     # w <= 1
    ]
    
    # Initial weights (start with zeros or random values)
    initial_weights = np.random.uniform(0, 1, A.shape[1])
    
    result = minimize(objective, initial_weights, constraints=constraints)
    optimized_weights = result.x
    
    # Predictions for all data (including unlabeled)
    predictions_all = all_data @ optimized_weights
    
    # Compute accuracy metrics based on labeled data
    predictions_labeled = A @ optimized_weights
    mse = mean_squared_error(M, predictions_labeled)
    mae = mean_absolute_error(M, predictions_labeled)
    r2 = r2_score(M, predictions_labeled)
    
    metrics = {"MSE": mse, "MAE": mae, "R²": r2}

    return optimized_weights, predictions_all, metrics


# Method 2: Random Forest ----------------------------------------------------

def random_forest(A, M, all_data, acronym, method="embeddings", test_size=0.2, random_state=42):
    A_train, A_test, M_train, M_test = train_test_split(A, M, test_size=test_size, random_state=random_state)
    
    model = RandomForestClassifier(
    n_estimators=200,        
    max_depth=5,            
    min_samples_leaf=5,    
    max_features='sqrt',    
    random_state=random_state
    )
    
    model.fit(A_train, M_train)
    feature_importances = model.feature_importances_

    # Predict on all_data
    proba = model.predict_proba(all_data)[:, 1]
    predictions = model.predict(A_test)
    accuracy = accuracy_score(M_test, predictions)
    
    return feature_importances, proba, accuracy

# Method 3: Neural Networks ---------------------------------------------------

def neural_network(A, M, all_data, acronym, method="embeddings", test_size=0.2, plot_path="accuracy_vs_epoch", epochs=250):
    A_train, A_test, M_train, M_test = train_test_split(A, M, test_size=test_size, random_state=42)
    
    if method=="embeddings":
        epochs = 150
        learning_rate_init=0.001
        max_iter=100
    else:
        epochs = 1000
        learning_rate_init=0.001
        max_iter=100
    
    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', max_iter=max_iter, warm_start=True, learning_rate_init=learning_rate_init, early_stopping=True)
    
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.fit(A_train, M_train)
        train_accuracy = model.score(A_train, M_train)
        test_accuracy = model.score(A_test, M_test)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    
    feature_importance = np.abs(model.coefs_[0]).sum(axis=1)
    all_proba = model.predict_proba(all_data)[:, 1]
    accuracy = model.score(A_test, M_test)
    return feature_importance, all_proba, accuracy

   
# Method 4: Support Vector Machines (SVM) ------------------------------------


def svm(A, M, all_data, acronym, method="embeddings"):
    A_train, A_test, M_train, M_test = train_test_split(A, M, test_size=0.2, random_state=42)
    
    #new
    svm_model = SVC(kernel="linear", C=1, probability=True)
    svm_model.fit(A_train, M_train)
    
    # ros = RandomOverSampler(random_state=42) # --> Tried this for UCSD
    # A_resampled, M_resampled = ros.fit_resample(A_train, M_train)
    # svm_model = SVC(kernel="linear", C=10, probability=True, class_weight='balanced')
    # svm_model.fit(A_resampled, M_resampled)

    M_pred = svm_model.predict(A_test)
    proba = svm_model.predict_proba(all_data)[:, 1]  # Use all_data for predictions
    accuracy = accuracy_score(M_test, M_pred)
    feature_importance = np.abs(svm_model.coef_).flatten()
    
    return feature_importance, proba, accuracy


# Method 5: Support Vector Machines Grid Search

def grid_search(A,M, all_data, acronym, method="embeddings"):

    # Split data into train & test sets
    A_train, A_test, M_train, M_test = train_test_split(A, M, test_size=0.2, random_state=42)


    param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto", 0.1]
    }
    
    grid_search = GridSearchCV(SVC(probability=True), param_grid)
    grid_search.fit(A_train, M_train)
    best_model = grid_search.best_estimator_
    M_pred = best_model.predict(A_test)
    proba = best_model.predict_proba(all_data)[:, 1]
    accuracy = accuracy_score(M_test, M_pred)

    # Feature Importance (SVM Coefficients)
    if hasattr(best_model, 'feature_importances_'):
        # If the model has the 'feature_importances_' attribute (e.g., RandomForest)
        feature_importance = best_model.feature_importances_

    elif hasattr(best_model, 'coef_'):
        # If the model has the 'coef_' attribute (e.g., SVC with kernel='linear', LogisticRegression)
        feature_importance = np.abs(best_model.coef_).flatten()
    
    else:
        # Use permutation importance for models without direct feature importance (e.g., SVC with non-linear kernel)
        result = permutation_importance(best_model, A_train, M_train, n_repeats=10, random_state=42)
        feature_importance = result.importances_mean
    

    return feature_importance, proba, accuracy


# Method 6: Logistic Regression -----------------------------------------------

def logistic_regression(A, M, all_data, acronym, method="embeddings"):
    A_train, A_test, M_train, M_test = train_test_split(A, M, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(A_train, M_train)
    
    weights = model.coef_[0]
    proba = model.predict_proba(all_data)[:, 1]  # Use all_data for predictions
    predictions = model.predict(A_test)
    
    accuracy = accuracy_score(M_test, predictions)
    
    return weights, proba, accuracy




