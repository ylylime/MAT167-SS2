import numpy as np
<<<<<<< HEAD:MAT167/Project_1/q2/both_ab_tested.py
import numpy
=======
import random
import scipy
from scipy.linalg import hilbert
>>>>>>> 73888be06e2b93f7a85051080d1c559ac826d07e:MAT167/q2/both_ab_tested.py

# checking functions I ask GPT to make for me ================================
def is_orthogonal(matrix):
    matrix = np.array(matrix)
    rows, cols = matrix.shape
    
    if rows != cols:
        return False
    
    # Check if the dot product of each pair of columns is close to zero
    for i in range(cols):
        for j in range(cols):
            dot_product = np.dot(matrix[:, i], matrix[:, j])
            if i == j:
                if not np.isclose(dot_product, 1.0):
                    return False
            else:
                if not np.isclose(dot_product, 0.0):
                    return False
    
    return True

def is_gram_schmidt_result(Q, A, tolerance=1e-5):
    Q = np.array(Q)
    A = np.array(A)
    
    if Q.shape != A.shape:
        return False
    
    m, n = A.shape
    
    # Check if Q has orthogonal unit columns
    for j in range(n):
        if not np.isclose(np.linalg.norm(Q[:, j]), 1.0, rtol=0, atol=tolerance):
            return False
        
        for i in range(j):
            if not np.isclose(np.dot(Q[:, i], Q[:, j]), 0.0, rtol=0, atol=tolerance):
                return False
    
    # Check if Q spans the same subspace as A
    for i in range(m):
        projection = np.dot(Q.T, A[i])
        if not np.all(np.isclose(A[i], np.dot(Q, projection), rtol=0, atol=tolerance)):
            return False
    
    return True



# CGS ================================================================
<<<<<<< HEAD:MAT167/Project_1/q2/both_ab_tested.py
import numpy
import math

def proj(v1, v2):
    # proj_vec = (numpy.dot(A[i], v_p) / numpy.dot(v_p, v_p)) * v_p
=======
def proj(v1, v2):
>>>>>>> 73888be06e2b93f7a85051080d1c559ac826d07e:MAT167/q2/both_ab_tested.py
    return (numpy.dot(v2, v1) / numpy.dot(v1, v1)) * v1

def normalize(i):
    normalized_vector = i / numpy.linalg.norm(i)
    return normalized_vector

def cgs(A):
    # Q_t is the transpose of final matrix / basis Q
    # store col vectors as row vectors in Q for easy modification
    Q_t = []
    
    for i in range(len(A)):
        # v_i 
<<<<<<< HEAD:MAT167/Project_1/q2/both_ab_tested.py
        a_i = numpy.transpose(numpy.transpose(A)[i])
        v_i = a_i
=======
        a_i = A[i]
        v_i = A[i]
>>>>>>> 73888be06e2b93f7a85051080d1c559ac826d07e:MAT167/q2/both_ab_tested.py
        
        # v_pt: previous v's transpose (row vector)
        for v_pt in Q_t:
             # v_p: previous v's  (col vector)
            v_p = numpy.transpose(v_pt)
            
            proj_vec = proj(v_p, a_i)
            v_i = v_i - proj_vec
            
        # append current v_i
        Q_t.append(numpy.transpose(v_i))
            
<<<<<<< HEAD:MAT167/Project_1/q2/both_ab_tested.py
    print(Q_t)
=======
    # print(Q_t)
>>>>>>> 73888be06e2b93f7a85051080d1c559ac826d07e:MAT167/q2/both_ab_tested.py
    
    for i in range(len(Q_t)):
        Q_t[i] = normalize(Q_t[i])
    
    Q = numpy.transpose(Q_t)
    return Q


<<<<<<< HEAD:MAT167/Project_1/q2/both_ab_tested.py


# test = numpy.array([[3.0, 1.0], [2.0, 2.0]])
test2 = numpy.array([[1.0, -1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
test3 = np.random.randint(-20, 21, size=(5, 5))


print(test3)
# print numpy.array(gs(test))
resultQ = cgs(test3)
print(resultQ)

print(is_orthogonal(resultQ))
print(is_gram_schmidt_result(resultQ, test3))

=======
>>>>>>> 73888be06e2b93f7a85051080d1c559ac826d07e:MAT167/q2/both_ab_tested.py
# MGS ===========================================================================
import numpy

def proj(v1, v2):
<<<<<<< HEAD:MAT167/Project_1/q2/both_ab_tested.py
    # proj_vec = (numpy.dot(A[i], v_p) / numpy.dot(v_p, v_p)) * v_p
=======
>>>>>>> 73888be06e2b93f7a85051080d1c559ac826d07e:MAT167/q2/both_ab_tested.py
    return (numpy.dot(v2, v1) / numpy.dot(v1, v1)) * v1

def normalize(i):
    normalized_vector = i / numpy.linalg.norm(i)
    return normalized_vector

def mgs(A):
    # Q_t is the transpose of final matrix / basis Q
    # use transpose of Q so each col vector becoms row vector and easier to access
<<<<<<< HEAD:MAT167/Project_1/q2/both_ab_tested.py
    Q_t = numpy.transpose(A).astype(float) # force it to be float
    print(Q_t)
=======
    Q_t = numpy.transpose(A)
>>>>>>> 73888be06e2b93f7a85051080d1c559ac826d07e:MAT167/q2/both_ab_tested.py
    
    for i in range(len(Q_t)):
        Q_t[i] = normalize(Q_t[i])
        q_i = numpy.transpose(Q_t[i])
        
        # subtract q_i's axis from rest q_k, k > i
        for k in range(i + 1, len(Q_t)):
            q_k = numpy.transpose(Q_t[k])
            
            # subtract projection, already normlized
            q_k = q_k - numpy.dot(q_i, q_k) * q_i
            
            Q_t[k] = numpy.transpose(q_k)
<<<<<<< HEAD:MAT167/Project_1/q2/both_ab_tested.py
    
    Q = numpy.transpose(Q_t)   
    return Q

# test = numpy.array([[3.0, 1.0], [2.0, 2.0]])
test2 = numpy.array([[1.0, -1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
test3 = np.random.randint(-20, 21, size=(5, 5))
print(test3)

resultQ = mgs(test3)
print(resultQ)

print(is_orthogonal(resultQ))
print(is_gram_schmidt_result(resultQ, test3))
=======
            
    Q = numpy.transpose(Q_t)   
    return Q


def main():

    # test = numpy.array([[3.0, 1.0], [2.0, 2.0]])
    test2 = numpy.array([[1.0, -1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    # print numpy.array(gs(test))
    resultcgs = cgs(test2)
    resultmgs = mgs(test2)

    print(resultcgs)
    # print(is_orthogonal(resultcgs))
    # print(is_gram_schmidt_result(resultcgs, test2))

    print(resultmgs)
    # print(is_orthogonal(resultmgs))
    # print(is_gram_schmidt_result(resultmgs, test2))


    n = random.randint(2,9)
    print(n)
    # random matrix of size n
    A = numpy.random.random((n, n)) 
    resultcgs = cgs(A) 
    resultmgs = mgs(A)
    
    # ||QtQ=I|| for cgs
    test2c1 = numpy.linalg.norm( numpy.dot(numpy.transpose(resultcgs), resultcgs) - numpy.identity(n))
    print(test2c1)
    # ||QtQ=I|| for mgs
    test2c1 = numpy.linalg.norm( numpy.dot(numpy.transpose(resultmgs), resultmgs) - numpy.identity(n))
    print(test2c1)
    
    
    # random matrix of size n following the given eq
    A = 0.00001 * numpy.eye(n) + scipy.linalg.hilbert(n)
    resultcgs = cgs(A)
    resultmgs = mgs(A)

    # ||QtQ=I|| for cgs
    test2c2 = numpy.linalg.norm( numpy.dot(numpy.transpose(resultcgs), resultcgs) - numpy.identity(n))
    print(test2c2)
    # ||QtQ=I|| for mgs
    test2c2 = numpy.linalg.norm( numpy.dot(numpy.transpose(resultmgs), resultmgs) - numpy.identity(n))
    print(test2c2)

main()
>>>>>>> 73888be06e2b93f7a85051080d1c559ac826d07e:MAT167/q2/both_ab_tested.py
