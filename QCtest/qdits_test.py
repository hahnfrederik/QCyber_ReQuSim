import pytest
import qdit_ext_matrix as d_mat
import requsim.libs.matrix as mat
import numpy as np
from cmath import sqrt

rng = np.random.default_rng()
d = rng.integers(2, high=7)
abs_tol = 1e-07

# test the transformation of |0>
def test_transform_0():
    z0 = d_mat.z_d(d, 0)
    transform = d_mat.Ha_d(d)
    x0 = transform @ z0
    x0 = d_mat.complex_isclose(x0)
    x0_t = np.zeros(d) * 1j
    for i in range(d):
        x0_t[i] = 1 / sqrt(d)
    x0_t = x0_t[:, np.newaxis]
    assert np.allclose(
        x0, x0_t, atol=abs_tol
    ), f"with dimension {d}, there is a error in transformation"


# test the transformation of a random eigenstate of Z
def test_transform_random():
    l = rng.integers(1, high=d)
    # print('testing the transformation of |',l,'>')
    zl = d_mat.z_d(d, l)
    transform = d_mat.Ha_d(d)
    xl = transform @ zl
    xl = d_mat.complex_isclose(xl)
    xl_t = np.zeros(d) * 1j
    for i in range(d):
        xl_t[i] = 1 / sqrt(d) * np.exp(1j * 2 * np.pi * l * i / d)
    xl_t = xl_t[:, np.newaxis]
    assert np.allclose(
        xl, xl_t, atol=abs_tol
    ), f"with dimension {d} and eigenstate {l}, there is a error in random transformation"


def test_transform_back_0():
    x0 = d_mat.x_d(d, 0)
    transform = d_mat.Ha_d(d)
    Ha_dagg = mat.H(transform)
    z0 = Ha_dagg @ x0
    z0 = d_mat.complex_isclose(z0)
    z0_t = np.zeros(d) * 1j
    z0_t[0] = 1
    z0_t = z0_t[:, np.newaxis]
    assert np.allclose(
        z0, z0_t, atol=abs_tol
    ), f"with dimension {d}, there is a error in back transformation"


def test_transform_back_random():
    l = rng.integers(1, high=d)
    # print('testing the tranformation of |', l,' hat>')
    xl = d_mat.x_d(d, l)
    transform = d_mat.Ha_d(d)
    Ha_dagg = mat.H(transform)
    zl = Ha_dagg @ xl
    zl = d_mat.complex_isclose(zl)
    zl_t = np.zeros(d) * 1j
    zl_t[l] = 1
    zl_t = zl_t[:, np.newaxis]
    assert np.allclose(
        zl, zl_t, atol=abs_tol
    ), f"with dimension {d} and eigenstate {l}, there is a error in random back transformation"


def test_fourier_twice():
    l = rng.integers(0, high=d)
    zl = d_mat.z_d(d, l)
    transform = d_mat.Ha_d(d)
    qd_X = d_mat.X_d(d)
    zl = qd_X @ transform @ zl
    zl = d_mat.complex_isclose(zl)
    zl_t = (np.exp(2 * np.pi * 1j / d) ** (-1 * l)) * d_mat.x_d(d, l)
    assert np.allclose(
        zl, zl_t, atol=abs_tol
    ), f"with dimension {d}, there is a error in doing X H operation on {l} eigenstate"


def test_ghz_d():
    n = rng.integers(0, high=5)
    ghz_d = d_mat.ghz_d(d, n)
    bs = rng.choice(2, n)
    B = np.sum(bs)
    t_Z = []
    for b in bs:
        if b == 0:
            t_Z.append(mat.I(d))
        else:
            t_Z.append(d_mat.Z_d(d))
    t_Z = mat.tensor(*t_Z)
    ghz_d_op = t_Z @ ghz_d
    omeg = np.exp(2 * np.pi * 1j / d)
    ghz_d_t = (
        1
        / sqrt(d)
        * np.sum(
            [omeg ** (i * B) * mat.tensor(*([d_mat.z_d(d, i)] * n)) for i in range(d)],
            axis=0,
        )
    )
    assert np.allclose(
        ghz_d_op, ghz_d_t
    ), f"with dimension {d} and {n} actors, there is a error when operating on GHZ"
