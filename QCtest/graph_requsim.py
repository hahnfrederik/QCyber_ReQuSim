import graphepp as gg
import numpy as np
from requsim.world import World
from requsim.quantum_objects import(
        Station,
        MultiSource,
        MultiQubit,
        SchedulingSource,
        MultiSchedulingSource,
    )


#extending graüh class with degree matrix and calculating laplacian and density matrix of a given graph
def deg_matrix(adj_matrix):
    N = adj_matrix.shape[0]
    deg = np.zeros((N,N), dtype=int)
    deg_sum = np.sum(adj_matrix, axis = 0)
    for i, degree in enumerate(deg_sum):
        deg[i,i] = degree
    return deg

class GraphsReq(gg.Graph)

    def __init__(self, N, E, sets=[]):
        super().__init__(N,E,sets)
        self.deg_mat = deg_matrix(self.adj())
        deg_sum = np.sum(deg_mat)
        self._rho = (self.deg_mat - self.adj())/deg_sum
    
    @property
    def rho(self):
        return self._rho


#12 qbit graph for quantum beaver triples
N = 12
b_graph_1 = GraphsReq(
            N=12, E=[(0,1), (1,2), (3,4),(4,5), (6,7), (2,8), (7,10), (5,9), (8,11), (19,11), (9,11)]
        )

b_graph_2_1 = GraphReq(


A = [0,4, 9, 10]
B= [1, 3, 7, 8]
R = [2, 5, 6, 11]
#initializing requsim


world = World()

# four points, so that there is no overlap
radiants = np.linspace(0,2*np.pi, 4)

def state_generation_1(source):
    return b_graph.rho

def stat_generation_2_1(source):
    return 

def time_distribution(source):
    comm_distance = max(
            [distance(source, t_station) for t_station in source.target_stations]
    )
    trial_time = 2* comm_distance /C
    eta = P_LINK * np.exp((-1) * comm_distance / L_ATT)
    num_trials = np.random.geometric(etq)
    time_taken = num_trials * trial_time


#creating stations
stations = []
for i in range(N):
    station = Station(
        world = world, position=np.array([np.cos(radiats[i]), np.sin(radiants[i])])
    )

source1 = MultiSchedulingSoure(
        world = world,
        position = np.array([0,0]),
        target_stations=stations,
        time_distribution=time_distribution,
        state_generation=state_generation,
    )

source2 = 


# basic 5 qubit ghz state

ghz_graph = GraphsReq(
            N=5, E=[(0,1),(0,2),(0,3),(0,4)]
        )


