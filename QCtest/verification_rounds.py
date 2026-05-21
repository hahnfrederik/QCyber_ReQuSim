#in order to import the development code
import sys
sys.path.insert(0,'../src/requsim/libs')

import numpy as np
import matrix as mat
import aux_functions as af
""" Verification protocol form paper Experimental Quantum Electronic Voting
"""

# function for ghz fidelity

def ghz_fidelity(rho: np.ndarray, num_parties):
    z0s = [mat.z0] * num_parties
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_parties
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)

    fidelity = np.real_if_close(
        np.dot(np.dot(mat.H(ghz_psi), rho), ghz_psi)[0, 0])
    
    return fidelity

#inserting noise channels manually 

def _x_noise_function(rho, epsilon):
    """A single-qubit bit-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.X, rho), mat.H(mat.X))


def _y_noise_function(rho, epsilon):
    """A single-qubit bit-and-phase-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.Y, rho), mat.H(mat.Y))


def _z_noise_function(rho, epsilon):
    """A single-qubit phase-flip channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    epsilon : scalar
        Error probability 0 <= epsilon <= 1.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    return (1 - epsilon) * rho + epsilon * np.dot(np.dot(mat.Z, rho), mat.H(mat.Z))


def _w_noise_function(rho, alpha):
    """A single-qubit depolarizing (white) noise channel.

    Parameters
    ----------
    rho : np.ndarray
        A single-qubit density matrix (2x2).
    alpha : scalar
        Error parameter alpha 0 <= alpha <= 1.
        State is fully depolarized with probability (1-alpha)

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    # trace is necessary if dealing with unnormalized states (e.g. in apply_single_qubit_map)
    return alpha * rho + (1 - alpha) * mat.I(2) / 2 * np.trace(rho)


def _ad_noise_function(rho, gamma):
    """ A single-qubit amplitude damping noise channel.

    Parameters
    ----------
    rho: np.darray
        A single_qubit density matrix (2x2).
    gamma : scalar
        amplitude damping probability 0<= gamma <=1.
        the excitation of the state decays with probability gamma.

    Returns
    -------
    np.ndarray
        The density matrix with the map applied.

    """
    K0 = np.diag([1, np.sqrt(1-gamma)])
    K1 = np.sqrt(gamma) * np.array([[0,1],[0,0]])

    return K0 @ rho @ mat.H(K0) + K1 @ rho @ mat.H(K1)







""" simulating measurement here.
Question is how to simulate (all measurements at once or measure each qubit?)
The question is not about the correctness, rather about efficency


Here, opting for measuring each qubit and then calculate the collapsed state

"""

def verify(rho, verifier = 0): 
    # not using verifier now, but in if it is necessary later
    N = int(np.log2(rho.shape[0]))
    
    # is there better way to generate angles
    angles = np.random.rand(1,N-1) * np.pi
    last_angle = np.pi - np.sum(angles) % np.pi    
    angles = np.append(angles, [last_angle])
    
    results = []
    
    for i in range(N):
        state_vec_1 = 1/np.sqrt(2)* (mat.z0 + np.exp(1j*angles[i]) * mat.z1)
        state_vec_2 = 1/np.sqrt(2)* (mat.z0 - np.exp(1j*angles[i]) * mat.z1)
    
        proj = []
        # compute projectors
        if i < N-1:
            proj.append(mat.tensor(state_vec_1 @ mat.H(state_vec_1),mat.I(2**(N-1-i))))
            proj.append(mat.tensor(state_vec_2 @ mat.H(state_vec_2), mat.I(2**(N-1-i))))
        else:
            proj.append(state_vec_1 @ mat.H(state_vec_1))
            proj.append(state_vec_2 @ mat.H(state_vec_2))
        
        #calculate probabilites
        probs = []
        #print(rho.shape)
        #print(i)
        p1 = np.trace(proj[0]@rho)
        p2 = np.trace(proj[1]@rho)
        if np.isclose(np.real(p1), 0):
            p1 = 0 + np.imag(p1) * 1j
        if np.isclose(np.real(p2), 0):
            p2 = 0 + np.imag(p2) * 1j
        p1 = np.real_if_close(p1)
        p2 = np.real_if_close(p2)
        assert np.imag(p1) == 0, p1 
        assert np.imag(p2) == 0, p2
        assert np.real(p1) >=0, p1
        assert np.real(p2) >= 0, p2
        probs.append(np.real(p1))
        probs.append(np.real(p2))
        
        #sanity check: probs[0] + probs[1] should be 1
        choice = np.random.choice(2,1, p=[probs[0], probs[1]])[0]
        results.append(choice)
    
    
        #some sanity checks for rho_new
        if i != 3:
            #collapse of state 
            rho_new = proj[choice] @ rho @ proj[choice] / probs[choice]
            rho = mat.ptrace(rho_new, [0])
    #print(results)
    #print((np.sum(angles)/np.pi)%2)
    return (np.sum(angles)/np.pi)%2 == np.sum(results)%2



#create quantum state that is epsilon far from ghz state as per paper
def eps_error(rho,N,noise_choice = 0,index=0,epsilon=0):
    if noise_choice == 0:  
        rho = af.apply_single_qubit_map(_x_noise_function, index, rho, epsilon)
    elif noise_choice == 1:
        rho = af.apply_single_qubit_map(_y_noise_function, index, rho, epsilon)
    elif noise_choice == 2:
        rho = af.apply_single_qubit_map(_z_noise_function, index, rho, epsilon)
    elif noise_choice == 3:
        rho = af.apply_single_qubit_map(_w_noise_function, index, rho, epsilon)
    else:
        rho = af.apply_single_qubit_map(_ad_noise_function, index, rho, epsilon)
    #print(epsilon)
    return rho


if __name__ == "__main__":
    N = 4
    
    #create ghz state
    rho = mat.ghz(N) @ mat.H(mat.ghz(N))
    
    print(ghz_fidelity(rho, N))
    #create state that is epsilon far from ghz
    
    #exponent for noise parameter
    exp = -1
    
    min_eps = 0.1
    
    t_eps = None

    noise_choice = 0

    while t_eps is None or t_eps > min_eps:
        rho = eps_error(rho, N, noise_choice, index = 0, epsilon = 10**exp)
        t_eps = np.sqrt(1-(ghz_fidelity(rho, N)**2))
        exp -= 1
    
    print(t_eps)
    for i in range(iterN):
        hits += verify(rho)
    print(hits/iterN)
