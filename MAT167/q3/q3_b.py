import numpy as np
import math

# =============== helper functions ===============
def random_invertible_matrix(rank, min_val = -20, max_val = 20):
    while True:
        # Generate a random matrix of size n x n
        random_matrix = np.random.randint(min_val, max_val + 1, size=(rank, rank))
        
        # Check if the matrix is invertible
        if np.linalg.matrix_rank(random_matrix) == rank:
            return random_matrix
        
def random_vector(rank, min_val = -20, max_val = 20):
    return np.random.randint(-20, 20 + 1, size=(rank, 1)), np.random.randint(-20, 20 + 1, size=(rank, 1))

def apply_rotation(matrix, cos, sin, axis_pos):
    # axis_pos used zero-outed row position
    row_a = matrix[axis_pos - 1].copy()
    row_b = matrix[axis_pos].copy()
    
    # calculate new rows
    # |  C S |
    # | -S C | for rotating clockwise ans zero out 2nd term
    
    matrix[axis_pos - 1] = cos * row_a + sin * row_b
    matrix[axis_pos] = - sin * row_a + cos * row_b


def apply_givens_rotation_zero_out(matrix, row, col):
    a = matrix[row - 1][col]
    b = matrix[row][col]
    r = math.sqrt(a*a + b*b)
    
    cos = a/r
    sin = b/r
    
    row_a = matrix[row - 1].copy()
    row_b = matrix[row].copy()
    
    matrix[row - 1] = cos * row_a + sin * row_b
    matrix[row] = - sin * row_a + cos * row_b
    
    return cos, sin

# =============== generate matrix and vector ===============
rank = 4
A = random_invertible_matrix(rank)
u, v = random_vector(rank)
v_t = np.transpose(v)
print(A)

# =============== Get original QR decom. ===============
Q, R = np.linalg.qr(A)

# print(Q)
# print(R)
# A_check = np.matmul(Q, R)
# print(A_check)
# print(np.matmul(u, v_t))

# =============== main solution body ===============
# calculate (Q^T)u as w
Q_T = np.transpose(Q)
w = np.matmul(Q_T, u)
print(w)
l2_norm = np.linalg.norm(w, ord=2)
print(l2_norm)

# part 1 rotation, rotate u to be [[r] [0] ... [0]]
rank = 4
Q_ht = Q_T.copy()
R_h = R.copy()
for i in range(rank - 1, 0, -1):
    # i is the row position of the term currently zeroing out
    a = w[i - 1][0]
    b = w[i][0]
    r = math.sqrt(a*a + b*b)
    
    cos = a/r
    sin = b/r
    
    # apply G (rotation matrix) to w (making sure rotation is correct
    # apply_rotation(w, cos, sin, i)
    # print(np.transpose(w))
    # tested working
    
    # rotate w
    w[i - 1][0] = r
    w[i][0] = 0
    
    # apply G (rotation matrix) to R_hat
    apply_rotation(R_h, cos, sin, i)
    apply_rotation(Q_ht, cos, sin, i)

print("\n")
print("Will be upper hesssenberg")
print(R_h)
print("\n\n")

# add norm(w)e1v_t to R_h
e1 = np.zeros((rank, 1))
e1[0][0] = 1
R_h = R_h + np.matmul(w[0][0] * e1, v_t)

# part 2 rotation, zero out the R_ht to be upper triangular
for i in range(1, rank):
    # zero out one term in R_h and save the rotation parameter c,s
    cos, sin = apply_givens_rotation_zero_out(R_h, i, i-1)
    
    # apply same rotation to Q_ht
    apply_rotation(Q_ht, cos, sin, i)


print(np.matmul(w[0][0] * e1, v_t))
# print result
print(R_h)
print(Q_ht)

# =============== check result ===============
Q_h = np.transpose(Q_ht)
print(Q_h)
A_hat_result = np.matmul(Q_h, R_h)

A_hat = A + np.matmul(u, v_t)

print(A_hat)
print(A_hat_result)