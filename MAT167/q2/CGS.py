import numpy

def gs_cofficient(v1, v2):
    return numpy.dot(v2, v1) / numpy.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def gs(X):
    Y = []
    res = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
        Y.append(temp_vec)
    return Y
    # normalize(Y)


def normalize(Y):
    for i in Y:
        normalized_vector = i / numpy.linalg.norm(i)
    return Y

# test = numpy.array([[3.0, 1.0], [2.0, 2.0]])
test2 = numpy.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])

# print numpy.array(gs(test))
print numpy.array(gs(test2))
