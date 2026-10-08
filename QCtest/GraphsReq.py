import graphepp as gg
import numpy as np
import requsim.libs.matrix as mat

# extending graph class with degree matrix and calculating laplacian and density matrix of a given graph


def int_to_bin(x, bits):
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])


int_bin = np.vectorize(int_to_bin, otypes=[np.ndarray])


def density_from_graph(adj_matrix):
    N = adj_matrix.shape[0]
    rho = np.ones((2**N, 2**N), dtype=np.complex64)
    # create 2**N possibilities in computational basis
    bins = np.arange(2**N)
    bins = int_bin(bins, N)
    bins = np.stack(bins)
    phases = (0.5 * np.diag(bins @ adj_matrix @ np.transpose(bins))) % 2
    U_g = np.diag(np.pow(-1, phases))
    return (1 / 2**N) * U_g @ rho @ U_g


def CZ(n, m, N):
    """gives the N-qubit CZ unitary acting on n-th and m-th qubit"""
    # construct unitary
    if n == m:
        raise ValueError("Nonsensical Input: CZ acts on two qubits")
    a = np.array([[1]])
    b = np.array([[1]])
    for i in range(N):
        if i == n:
            a = mat.tensor(a, mat.z0 @ mat.H(mat.z0))
            b = mat.tensor(b, mat.z1 @ mat.H(mat.z1))
        elif i == m:
            a = mat.tensor(a, mat.I(2))
            b = mat.tensor(b, mat.Z)
        else:
            a = mat.tensor(a, mat.I(2))
            b = mat.tensor(b, mat.I(2))
    return a + b


def graph_state(N, E):
    """Return the graph state in the computational basis.

    Parameters
    ----------
    graph : nx.Graph
        The graph describing the graph state.

    Returns
    -------
    np.ndarray
        A column-vector of the graph state given in the computational basis.
        shape = (2**N, 1)

    """
    aux = [mat.x0] * N
    psi = mat.tensor(*aux)
    for edge in E:
        psi = CZ(edge[0], edge[1], N) @ psi
    return psi


def transform_matrix(N, psi):  # only works for pure graph state vector
    my_tuple = ()
    for i in range(2**N):
        operator = np.array([[1]])
        for n in range(N):
            if i & (1 << ((N - 1) - n)):
                operator = mat.tensor(operator, mat.Z)
            else:
                operator = mat.tensor(operator, mat.I(2))
        print(operator.shape)
        my_tuple += (np.dot(operator, psi),)
    return mat.H(np.hstack(my_tuple))


class GraphsReq(gg.Graph):

    def __init__(self, N, E, sets=[]):
        super().__init__(N, E, sets)
        self._rho = density_from_graph(self.adj)
        self._psi = graph_state(N, E)
        self._t_matrix = transform_matrix(self.N, self.psi)
        # self._rho_graph = (t_matrix @ self.rho) @ mat.H(t_matrix)
        self._rho_graph = self.psi @ mat.H(self.psi)

    @property
    def rho(self):
        return self._rho

    @property
    def rho_graph(self):
        return self._rho_graph

    @property
    def psi(self):
        return self._psi

    @property
    def t_matrix(self):
        return self._t_matrix
