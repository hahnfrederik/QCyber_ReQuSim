#in order to import the development code
import sys
import os
sys.path.insert(0,'../src/requsim/libs')

import numpy as np
#import matrix as mat
import requsim.libs.matrix as mat
#import aux_functions as af
import requsim.libs.aux_functions as af
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
    # not using verifier now, but inserted for later
    N = int(np.log2(rho.shape[0]))
    
    # # is there better way to generate angles
    # angles = np.random.rand(1,N-1) * np.pi
    # last_angle = np.pi - np.sum(angles) % np.pi   
    # angles = np.append(angles, [last_angle])
    
    ### alternative suggestion:
    angles = np.random.rand(N - 1) * np.pi
    # pick parity m uniformly from {0, 1} and then make sure the sum is m*pi
    m = np.random.randint(0, 2)  # or any integer really, but parity is what matters
    last_angle = (m * np.pi - np.sum(angles)) % np.pi 
    angles = np.append(angles, last_angle)

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
        if i != 3: # N-1 instead of 3?
            #collapse of state 
            rho_new = proj[choice] @ rho @ proj[choice] / probs[choice]
            rho = mat.ptrace(rho_new, [0])
    #print(results)
    #print((np.sum(angles)/np.pi)%2)
    # return (np.sum(angles)/np.pi)%2 == np.sum(results)%2 

    ### I think it is safer to make sure that both sides are integers so maybe:
    expected_parity = int(round(np.sum(angles) / np.pi)) % 2
    measured_parity = int(np.sum(results) % 2)
    return expected_parity == measured_parity



#create quantum state that is epsilon far from ghz state as per paper
def eps_error(rho,N,noise_choice = 0,index=0, error=0):
    if noise_choice == 0:  
        rho = af.apply_single_qubit_map(_x_noise_function, index, rho, error)
        noise = 'x_noise'
    elif noise_choice == 1:
        rho = af.apply_single_qubit_map(_y_noise_function, index, rho, error)
        noise = 'y_noise'
    elif noise_choice == 2:
        rho = af.apply_single_qubit_map(_z_noise_function, index, rho, error)
        noise = 'z_noise'
    elif noise_choice == 3:
        # for w_noise we do one minus, since a the error probability is reversed in comparison to others
        rho = af.apply_single_qubit_map(_w_noise_function, index, rho, 1-error)# Maybe for consistency we should rewrite the _w_noise_function
        noise = 'w_noise'
    else:
        rho = af.apply_single_qubit_map(_ad_noise_function, index, rho, error)
        noise = 'ad_noise'
    #print(epsilon)
    return rho, noise


if __name__ == "__main__":
    N = 4
    
    #create ghz state
    rho = mat.ghz(N) @ mat.H(mat.ghz(N))
    #create state that is epsilon far from ghz

    min_eps = 0.1
    
    t_eps = None
    
    for noise_choice in range(5):
        #exponent for error parameter
        exp = -1
        # better name? variable to get a more fine tuned exponent
        step = 0
        while t_eps is None or not np.isclose(t_eps, min_eps):
            rho2, noise_string = eps_error(rho, N, noise_choice, index = 0, error = 10**exp)
            t_eps = np.sqrt(1-(ghz_fidelity(rho2, N)**2)) # I think the square should just be correct if the ghz_fidelity function would output the square root of the overlap. But to me it looks like ⟨GHZ∣ρ∣GHZ⟩ is clalculated. So either we remove the square here or add a square root to the fidelity function.
            if t_eps > min_eps:
                exp -= 1* (10**step)
            if t_eps < min_eps:
                exp += 1 * (10**step)
                step -= 1
                exp -= 1 * (10**step)
        

        # make iterN dependent on the probability
        p = (t_eps**2)/4
        iterN = int(15/p)
        print('-----------------------------------')
        print('the noise being used is', noise_string)
        print('the state being used is', t_eps, 'far from the ghz state')
        print('theoretical minimum probability of state failing verification protocol:', p)

        hits = 0
        print('using', iterN, 'iterations in simulation')
        for i in range(iterN):
            hits += verify(rho2)
        print('simulation probability of the state failing verification:', 1-hits/iterN)    
        t_eps = None


