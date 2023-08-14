import numpy


def modified_gs(A:numpy.ndarray)->numpy.ndarray:
    num_vecs = A.shape[0]
    num_dims = A.shape[1]
    
    L = numpy.zeros(num_vecs)
    for i in range(num_vecs):
        L[i] = numpy.sqrt(A[i].T@A[i])
    
    
    V = A.copy() / L
    B = V.copy()   
    for j in range(0, num_vecs):
        B[j] = V[j]/numpy.sqrt(V[j].T@V[j])
        for k in range(j, num_vecs):
            V[k] = V[k] - (B[j].T@V[k])*B[j]    
    return B
    