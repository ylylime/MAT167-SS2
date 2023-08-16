import numpy

def proj(v1, v2):
    return (numpy.dot(v2, v1) / numpy.dot(v1, v1)) * v1

def cgs(X):
    Y = []
    res = X[i]
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            # proj_vec = (numpy.dot(X[i], inY) / numpy.dot(inY, inY)) * X[i]
            proj_vec = proj(inY, X[i])
            Y.append(proj_vec)

        for inW in range(0, len(Y)-1):
            res -= Y[inW+1]
        return res
        print(normalize(res))
    return Y

def normalize(i):
    normalized_vector = i / numpy.linalg.norm(i)
    return normalized_vector



# test = numpy.array([[3.0, 1.0], [2.0, 2.0]])
test2 = numpy.array([[1.0, -1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 2.0]])

# print numpy.array(gs(test))
print numpy.array(cgs(test2))
