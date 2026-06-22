import requsim.libs.matrix as mat
import numpy as np
from cmath import sqrt


# extending the elementary vectors and operators for d dimensions


def z_d(d, i):
    # here d is for dimensions and i is for the basis element
    state_vec = np.zeros(d)
    state_vec[i] = 1
    return state_vec.reshape(d, 1)


def x_d(d, i):
    state_vec = z_d(d, i)


# define the generlaized Hadarmard for qudits, which is the quantum foureier transform
def Ha_d(d):
    omeg = np.exp(2 * np.pi * 1j / d)
    Ha_mat = np.zeros((d, d)).astype(np.complex64)
    for i in range(d):
        for k in range(d):
            Ha_mat[i][k] = omeg ** (i * k)
    return np.asmatrix(Ha_mat / sqrt(d))


if __name__ == "__main__":
    a = z_d(4, 0)
    print(a)
    b = Ha_d(4) @ a
    print(b)
    print(Ha_d(4).H @ b)
