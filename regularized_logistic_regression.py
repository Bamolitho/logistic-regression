import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import *
import os

def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost 
    """

    m, n = X.shape
    
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    
    # You need to calculate this value
    reg_cost = 0.
    
    for j in range(n):
        reg_cost += (lambda_/(2*m))*(w[j]**2)
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost

    return total_cost

def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for logistic regression with regularization
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    m, n = X.shape
    
    dj_db_without_reg, dj_dw_without_reg = compute_gradient(X, y, w, b)
    dj_db_final = dj_db_without_reg

    dj_dw_final = np.zeros(w.shape)
    for j in range(n): 
        dj_dw_final[j] = dj_dw_without_reg[j] + (lambda_/m)*w[j]
    
        
    return dj_db_final, dj_dw_final

X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ 
lambda_ = 0.01    

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m = X.shape[0]   
    p = np.zeros(m)
   
    # Loop over each example
    for i in range(m):   
        z_wb = np.dot(w, X[i]) + b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)


        # Apply the threshold
        p[i] = int(f_wb >= 0.5)
        
    return p

# Plotting the decision boundary
def descision_boundary():
    plot_decision_boundary(w, b, X_mapped, y_train)
    # Set the y-axis label
    plt.ylabel('Microchip Test 2') 
    # Set the x-axis label
    plt.xlabel('Microchip Test 1') 
    plt.legend(loc="upper right")
    plt.show()


#Compute accuracy on the training set
def compute_accuracy():
    p = predict(X_mapped, w, b)
    print(f'Train Accuracy: {np.mean(p == y_train) * 100:.2f} %')

def user_input_prediction(w, b):
    """
    Prompt user for two test results and output prediction
    """
    print("\n=== Predict Microchip Quality ===")
    try:
        test1 = float(input("Enter result of Microchip Test 1 ((entre environ -1 et 1.5)): "))
        test2 = float(input("Enter result of Microchip Test 2 (entre environ -1 et 1.2): "))
        user_features = map_feature(np.array([test1]), np.array([test2]))
        result = predict(user_features, w, b)
        print("Verdict:", "Pass ✅" if result == 1 else "Reject ❌")
    except Exception as e:
        print("Invalid input. Error:", e)

