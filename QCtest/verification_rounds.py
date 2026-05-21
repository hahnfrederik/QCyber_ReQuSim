#in order to import the development code
import sys
sys.path.insert(0,'../src/requsim/libs')

import numpy as np
import matrix as mat
""" Verification protocol form paper Experimental Quantum Electronic Voting
"""

#is there a better way to generate angles?






""" simulating measurement here.
Question is how to simulate (all measurements at once or measure each qubit?)
The question is not about the correctness, rather about efficency


Here, opting for measuring each qubit and then calculate the collapsed state

"""

def verify(rho, verifier = 0): 
    # not using verifier now, but in if it is necessary later
    N = int(np.log2(rho.shape[0]))
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
def eps_error(epsilon):
   '''
   haven done anything yet
   '''
   return None


if __name__ == "__main__":
    iterN = 10
    hits = 0
    N = 4
    rho = mat.ghz(N) @ mat.H(mat.ghz(N))
    for i in range(iterN):
        hits += verify(rho)
    print(hits/iterN)



