import numpy as np

A = np.array([[1, 2, 3, 4, 5],
     [2, 4, 5, 6, 6],
     [2, 4, 5, 6, 6],
     [2, 4, 5, 6, 6],
     [2, 4, 5, 6, 6],
     [2, 4, 5, 6, 6],
     [2, 4, 5, 6, 6],
     [2, 3, 5, 6, 6]])

print(A)
print(A.shape)
print(A[1,2])
print(len(A))

flat = A.flatten()
print(flat)
print(len(flat))

B = np.full(len(A), 4)
B[::2] = 5
print(B)

print(A[::2])