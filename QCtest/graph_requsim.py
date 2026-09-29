import graphepp as gg
import numpy as np
from requsim.world import World
from requsim.quantum_objects import (
    Station,
    MultiSource,
    MultiQubit,
    SchedulingSource,
    MultiSchedulingSource,
)
import requsim.libs.matrix as mat
from requsim.libs.aux_functions import distance
from requisim.tools.protocol import Protocol

# some constants
C = 2e8
P_LINK = 0.80
L_ATT = 22e3


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


class GraphsReq(gg.Graph):

    def __init__(self, N, E, sets=[]):
        super().__init__(N, E, sets)
        self._rho = density_from_graph(self.adj)

    @property
    def rho(self):
        return self._rho


# 12 qbit graph for quantum beaver triples
N = 12
b_graph_1 = GraphsReq(
    N=12,
    E=[
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (6, 7),
        (2, 8),
        (7, 10),
        (5, 9),
        (8, 11),
        (19, 11),
        (9, 11),
    ],
)

# scenario 1
# ghz state sending out


A = [0, 4, 9, 10]
B = [1, 3, 7, 8]
R = [2, 5, 6, 11]
# initializing requsim


stationN = 4
world = World()

# +1 points, so that there is no overlap
radiants = np.linspace(0, 2 * np.pi, stationN + 1)


def state_generation_scenario_1_1(source):
    ghz_state = mat.ghz(stationN) @ mat.H(mat.ghz(stationN))
    return ghz_state


def time_distribution_scenario_1_1(
    source,
):  # we generalize here by using the longest distance for all
    comm_distance = max(
        [distance(source, t_station) for t_station in source.target_stations]
    )
    trial_time = 2 * comm_distance / C
    eta = P_LINK * np.exp((-1) * comm_distance / L_ATT)
    num_trials = np.random.geometric(eta)
    time_taken = num_trials * trial_time
    return time_taken


# different distribution choices


def scenario_1_1(source_main):
    # in this scanario, we purify the state before (increase the fidelity locally at the source)
    source_main.schedule_event()
    world.print_status()
    while world.event_queue.next_event is not None:
        world.event_queue.resolve_next_event()


def scenario_1_2(source_main):
    source_main.schedule_event()
    source_main.schedule_event()

    world.print_status()
    while world.event_queue.next_event is not None:
        world.event_queue.resolve_next_event()
        world.print_status()


def scanario_1_3():
    bell_sources = []
    current_message = None

    return


# two different TCP implementations. One based on graph state basis and on ein the computational basis

# for graph state basis, we switch from computational to graph state basis and then we use the tcp protocol on it
# all the errors that we want to put on the state should be also translated to the graph sate basis.

# this code is taken form julius

# the graph should be of type GraphsReq


class TCP_graph_protocol(Protocol):

    def __init__(self, graph, world=None):
        super().__init__(world)
        self.graph = graph
        self.U = self.transform_matrix()

    def transform_matrix():
        my_tuple = ()
        for i in range(2**self.graph.N):
            operator = np.array([[1]])
            for n in range(self.graph.N):
                if i & (1 << ((N - 1) - n)):
                    operator = mat.tensor(operator, mat.Z)
                else:
                    operator = mat.tensor(operator, mat.I(2))
            my_tuple += (np.dot(operator, self.graph.rho),)
        return mat.H(np.hstack(my_tuple))

    def setup(self, world=None, communication_speed=None):
        """
        create the sources and so on

        """
        # creating stations
        self.stations = []
        for i in range(self.graph.N):
            station = Station(
                world=self.world,
                position=np.array([np.cos(radiants[i]), np.sin(radiants[i])]),
            )
            self.stations += [station]

        # creating source
        self.source = MultiSchedulingSource(
            world=self.world,
            position=np.array([0, 0]),
            target_stations=self.stations,
        )

    def check(self, message=None):
        """check current status and schedule new events.

        looks globally at the status of the whole 'world' and decides which events need to be scheduled.
        """
        # sending qubits
        # scenario 1
        if message == "send_1":
            self.source_1.schedule_event()
        if message == "send_2":
            return
        # purifying central
        if message == "purify_central":
            return
        # purify outer
        if message == "purify_outer":
            return


# for computational basis,...

if __name__ == "__main__":
    scenario_1_2()
