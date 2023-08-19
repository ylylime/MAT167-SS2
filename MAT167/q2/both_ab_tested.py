import numpy as np
import numpy
import random

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
def proj(v1, v2):
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
        a_i = A[i]
        v_i = A[i]
        
        # v_pt: previous v's transpose (row vector)
        for v_pt in Q_t:
             # v_p: previous v's  (col vector)
            v_p = numpy.transpose(v_pt)
            
            proj_vec = proj(v_p, a_i)
            v_i = v_i - proj_vec
            
        # append current v_i
        Q_t.append(numpy.transpose(v_i))
            
    print(Q_t)
    
    for i in range(len(Q_t)):
        Q_t[i] = normalize(Q_t[i])
    
    Q = numpy.transpose(Q_t)
    return Q


# MGS ===========================================================================
import numpy

def proj(v1, v2):
    return (numpy.dot(v2, v1) / numpy.dot(v1, v1)) * v1

def normalize(i):
    normalized_vector = i / numpy.linalg.norm(i)
    return normalized_vector

def mgs(A):
    # Q_t is the transpose of final matrix / basis Q
    # use transpose of Q so each col vector becoms row vector and easier to access
    Q_t = numpy.transpose(A)
    
    for i in range(len(Q_t)):
        Q_t[i] = normalize(Q_t[i])
        q_i = numpy.transpose(Q_t[i])
        
        # subtract q_i's axis from rest q_k, k > i
        for k in range(i + 1, len(Q_t)):
            q_k = numpy.transpose(Q_t[k])
            
            # subtract projection, already normlized
            q_k = q_k - numpy.dot(q_i, q_k) * q_i
            
            Q_t[k] = numpy.transpose(q_k)
            
    Q = numpy.transpose(Q_t)   
    return Q


def main():

    # test = numpy.array([[3.0, 1.0], [2.0, 2.0]])
    test2 = numpy.array([[1.0, -1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
    # print numpy.array(gs(test))
    resultcgs = cgs(test2)
    resultmgs = mgs(test2)

    print(resultcgs)
    print(is_orthogonal(resultcgs))
    print(is_gram_schmidt_result(resultcgs, test2))

    print(resultmgs)
    print(is_orthogonal(resultmgs))
    print(is_gram_schmidt_result(resultmgs, test2))


    n = random.randint(1,9)
    print(n)
    A = numpy.random.random((n, n)) 
    test2c1 = numpy.linalg.norm( numpy.dot(numpy.transpose(A), A) - numpy.identity(n))

    resultcgs = cgs(test2c1)
    resultmgs = mgs(test2c1)

    # ||QtQ=I||
    A = 0.00001 * numpy.eye(n) + scipy.linalg.hilbert(n)
    test2c2 = numpy.linalg.norm( numpy.dot(numpy.transpose(A), A) - numpy.identity(n))

    resultcgs = cgs(test2c2)
    resultmgs = mgs(test2c2)


main()