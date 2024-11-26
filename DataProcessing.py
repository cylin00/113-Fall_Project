import wave
import os
import numpy as np

def GetMaxAmplitude(n0, data):
    num_chunks = len(data) // n0
    reshaped_data = data[:num_chunks * n0].reshape(-1, n0)
    
    A = np.max(reshaped_data, axis=1)
    
    rho = 0
    A[A < rho] = 0
    
    return A


# def GetMaxAmplitude(n0, data):

#     A = np.zeros(len(data) // n0)

#     for idx, i in enumerate(range(0, len(data), n0)):
#         group = data[i:i + n0]
#         A[idx] = max(group)  

#     rho = 0
#     for i in range(0, len(A)):
#         if A[i] < rho:
#             A[i] = 0

#     return A

def NormalizeData(A):

    mean_A = np.mean(A)

    for i in range(0, len(A)):
        A[i] = A[i] / (0.2 + 0.1 * mean_A)

    return A

def FractionalData(A, l):

    for i in range(0, len(A)):
        A[i] = A[i] ** l  

    return A

def EnvelopData(B, F):

    Enveloped_Data = np.zeros(len(B))

    for i in range(0, len(B)):
        for j in range(0, len(F) - 1):
            Enveloped_Data[i] += B[i - j] * F[j]

    return Enveloped_Data