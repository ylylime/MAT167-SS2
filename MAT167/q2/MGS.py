import numpy

def proj(v1, v2):
    return (numpy.dot(v2, v1) / numpy.dot(v1, v1)) * v1

def mgs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec -= proj_vec
            # temp_vec = normalize(temp_vec)
            print(normalize(temp_vec))
        Y.append(temp_vec)
    return Y

def normalize(i):
    normalized_vector = i / numpy.linalg.norm(i)
    return normalized_vector

# test = numpy.array([[3.0, 1.0], [2.0, 2.0]])
test2 = numpy.array([[1.0, -1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])

# print numpy.array(gs(test))
print numpy.array(mgs(test2))



# def modified_gs(A:numpy.ndarray)->numpy.ndarray:
#     num_vecs = A.shape[0]
#     num_dims = A.shape[1]
    
#     L = numpy.zeros(num_vecs)
#     for i in range(num_vecs):
#         L[i] = numpy.sqrt(A[i].T@A[i])
    
    
#     V = A.copy() / L
#     B = V.copy()   
#     for j in range(0, num_vecs):
#         B[j] = V[j]/numpy.sqrt(V[j].T@V[j])
#         for k in range(j, num_vecs):
#             V[k] = V[k] - (B[j].T@V[k])*B[j]    
#     return B
