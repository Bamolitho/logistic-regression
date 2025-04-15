# logistic-regression-project

Project Description: Logistic Regression with Regularization for binary classification
This project implements Logistic Regression with regularization to predict binary classification based on two input features. The model is trained using a dataset of 100 samples with two features (x1 and x2), each associated with a binary label (0 or 1).

Key Components:
Logistic Regression: A classification algorithm used to predict the probability of a binary outcome based on input features.

Regularization: This technique helps prevent overfitting by adding a penalty term to the cost function. The project uses L2 regularization (Ridge Regularization).

Cost Function: The cost function includes both the usual logistic loss and the regularization term. It is used to measure how well the model fits the training data.

Gradient Descent: An optimization technique to minimize the cost function by adjusting the model's parameters (weights).

User Input: The program allows users to input two feature values (x1, x2) and predict the class label (0 or 1) using the trained model.

Workflow:
Dataset: The dataset is loaded from a CSV file containing two input features and a binary target variable.

Model Training: The logistic regression model is trained using gradient descent with regularization, optimizing the weights to minimize the cost function.

Cost History: The cost during each iteration of gradient descent is recorded and can be plotted for visualization.

Prediction: After training, users can input two feature values, and the program will predict whether the output is 0 or 1 based on the learned model.

Evaluation: The program computes the training accuracy by comparing the predicted labels to the actual labels.

Functionality:
Train a logistic regression model with regularization.

Plot the boundary line of the training set
Print the   accuray of the training set

Allow users to input data and predict the class label.

Provide options to clear the console and exit the program.

Requirements:
Python 3.x

Libraries: numpy, matplotlib, pandas (for data handling), os (for clearing the console    

Source: This project was completed as part of the Machine Learning Specialization, specifically in Course 1/3, Week 3.

It demonstrates essential machine learning concepts, including logistic regression, gradient descent, regularization, and model evaluation, providing hands-on experience in building and deploying machine learning models.
