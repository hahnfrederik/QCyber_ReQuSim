import numpy as np


def angles(N):
    """Return a list of angles, such that the sum of the angles is a multiple of pi

    Parameters
    ----------
    N : integer
        the amount angles we want in the list

    Returns
    -------
    m: {0,1}
     the parity of the multiple of pi that the angles sum up to

    list: np.ndarrays
        the angles [0, pi] that sum up to a whole multiple of pi
    """

    # Kinda weird right now, since m is irrelevant for the last angle calculation.
    # But if it is irrelevnat, than how can it be the parity of the multiple of the sum of the angles?

    angles = np.random.rand(N - 1) * np.pi
    # pick parity m uniformly from {0, 1} and then make sure the sum is m*pi
    m = np.random.randint(0, 2)  # or any integer really, but parity is what matters
    last_angle = (
        m * np.pi - np.sum(angles)
    ) % np.pi  # @Jan: I think here we can actually just do -np.sum(angles) % np.pi
    angles = np.append(angles, last_angle)
    return m, angles


def ghz_fidelity(rho: np.ndarray, N):
    """function to calculate the fidelity of a given density matrix to the ghz state

    Parameters
    ----------
    rho: np.ndarray
        the density matrix of the quantum state that is to be compared to the ghz state
    N: integer
        the number of qubits in the system described by rho
        Theoretically, it is not needed here since this can be calculated by rho itself

    Returns
    -------
    fidelity: scalar
        the fidelity of the state rho and a ghz state
    """

    # Generalize the function for general fidelity function for two states
    z0s = [mat.z0] * num_parties
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_parties
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)

    fidelity = np.real_if_close(np.dot(np.dot(mat.H(ghz_psi), rho), ghz_psi)[0, 0])

    return fidelity
