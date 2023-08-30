import numpy as np
from scipy.optimize import approx_fprime
import math
import matplotlib.pyplot as plt

def slove_linear_with_svd(A, b):
    U, s, V_t = np.linalg.svd(A)
    S = np.diag(s)
    
    # Calculate the pseudo-inverse of S
    S_inv = np.linalg.pinv(S)
    
    # A = USV_t
    # A^-1 = V S^-1 U_t
    # Calculate the solution using the pseudo-inverse
    x = (V_t.T) @ S_inv @ (U.T) @ b
    
    return x

# Example usage
A = np.array([[2, 1, 1],
              [1, 3, 3],
              [4, 1, 1]])
b = np.array([3, 5, 7])

solution = slove_linear_with_svd(A, b)
print("Solution:", solution)

# ==========================================================

def next_step_vector_classical(f, x_n):
    # calculate Jacobian at x_n
    J = approx_fprime(x_n, f, epsilon=1e-6)
    
    # Solve the linear system J * delta_x = fx for the Newton step
    delta_x = solve_linear_system_svd(J, function(x_n))
        
    # Update the solution
    x_next = x_n - delta_x
    
    return x_next, delta_x
    
    
def newtonsMethod(function, x_0, epsilon_x = 1e-6, epsilon_f = 1e-6, max_itr = 10000):
    x_n = x_0
    f_n = function(x_n)
    
    if np.linalg.norm(f_n) <= epsilon_f:
        return x_n
    
    itr = 0
    delta_x = 1000
    residuals = [np.linalg.norm(f_n)]
    while (itr < max_itr):
        x_n, delta_x = next_step_vector_classical(function, x_n)
        f_n = function(x_n)
        
        if np.linalg.norm(delta_x) <= epsilon_x or np.linalg.norm(f_n) <= epsilon_f:
            break
            
        # Calculate and store the norm of the residual
        residual_norm = np.linalg.norm(f_n)
        residuals.append(residual_norm)
        print(f_n)
    
    return x_n, residuals

def calculate_rate_of_convergence(residuals):
    roc = []
    
    i = 0
    while (i + 3 < len(residuals)):
        top_log = math.log(abs((residuals[i + 3] - residuals[i + 2])/(residuals[i + 2] - residuals[i + 1])))
        bottom_log = math.log(abs((residuals[i + 2] - residuals[i + 1])/(residuals[i + 1] - residuals[i])))
        
        q = top_log/bottom_log
        roc.append(q)
        
        i += 1
        
    return roc
# ==========================================================

def plot_rate_of_convergence(roc):
    indices = np.arange(len(roc))
    # Create the plot
    plt.plot(indices, roc)

    # Add labels and title
    plt.xlabel("Index")
    plt.ylabel("Rate")
    plt.title("Rate of Convergence")

    # Show the plot
    plt.show()

def function(vector_x):
    x = vector_x[0]
    y = vector_x[1]
    return np.array([y - 2*x*y,
                     2*x*y - y])

# Initial guess
x_0 = np.array([9000.0, -9000.0])

# Call the newtonMethod function
x_result, residuals = newtonsMethod(function, x_0, 1e-32, 1e-32)

print("done")
print(function(x_result))
print(residuals)

rate_of_convergence = calculate_rate_of_convergence(residuals)
print("done")

print(rate_of_convergence)
plot_rate_of_convergence(rate_of_convergence)
# ==========================================================