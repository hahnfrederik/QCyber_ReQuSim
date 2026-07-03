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
    return Ha_d(d) @ state_vec


def complex_isclose(a):
    reals = np.real(a)
    reals = np.multiply(np.isclose(reals, 0) == False, reals)
    imags = np.imag(a)
    imags = np.multiply(np.isclose(imags, 0) == False, imags)
    return reals + imags * 1j


# define the generlaized Hadarmard for qudits, which is the quantum foureier transform
def Ha_d(d):
    omeg = np.exp(2 * np.pi * 1j / d)
    Ha_mat = np.zeros((d, d)).astype(np.complex64)
    for i in range(d):
        for k in range(d):
            Ha_mat[i][k] = omeg ** (i * k)
    return Ha_mat / sqrt(d)


# generalized Pauli Z matrix for qudits with optional argument x (as in Wildes book, which is equivalent to multiple applications of the operation)
def Z_d(d, x=1):
    diag = np.zeros(d).astype(np.complex128)
    omeg = np.exp(2 * np.pi * x * 1j / d)
    for i in range(d):
        diag[i] = omeg**i
    return np.diag(diag).astype(np.complex128)


# generalized Pauli X matrix, analogous to above
def X_d(d, x=1):
    ret = np.zeros((d, d))
    for i in range(d):
        ret[(i + x) % d, i] = 1
    return ret


def ghz_d(d, n):
    state_vecs = [mat.tensor(*([z_d(d, i)] * n)) for i in range(d)]
    return 1 / sqrt(d) * sum(state_vecs)


if __name__ == "__main__":
    a = ghz_d(3, 2)
    print(a @ mat.H(a))
