#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np

# Method a)
def qr_update(Q, R, u, v):
    
    Qt_u = np.dot(Q.T, u)

    R += np.outer(Qt_u, v)

    Q_new, R_new = np.linalg.qr(R)
    
    return Q @ Q_new, R_new

# Generate data
n = 4
A = np.random.rand(n, n)
Q, R = np.linalg.qr(A)
u = np.random.rand(n)
v = np.random.rand(n)

# QR decomposition using method a)
Q_updated, R_updated = qr_update(Q, R, u, v)

# naively computing QR decomposition 
A_updated = A + np.outer(u, v)
Q_naive, R_naive = np.linalg.qr(A_updated)

# Compare the results
print("Q (naive) - Q (updated):")
print(Q_naive - Q_updated)
print("\nR (naive) - R (updated):")
print(R_naive - R_updated)

