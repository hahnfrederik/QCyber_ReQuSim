"""
Verification protocol using requsim
"""

from requsim.world import World
from requsim.quantum_objects import (
    Station,
    SourceMult,
    MultiQubit,
    SchedulingSource,
    MultiSchedulingSource,
)
import requsim.libs.matrix as mat
import numpy as np
from requsim.events import MultiSourceEvent, MeasurementEvent
from requsim.libs.aux_functions import distance
from verification_rounds import VerifyProtocol
import QC_aux_functions as qaf

C = 2e8  # speed of light

N = 10  # number of users
P_LINK = 0.80
T_DP = 100e-3
LAMBDA_BSM = 0.99
L_ATT = 22e3

speedMeas = 1e-9

# put stations around source in a circle
radius = 2e3

# N+1 since we dont want the first and last to overlap
radiants = np.linspace(0, 2 * np.pi, N + 1)


def state_generation(source):
    ghz_state = mat.ghz(N) @ mat.H(mat.ghz(N))
    return ghz_state


def time_distribution(source):
    comm_distance = max(
        [distance(source, t_station) for t_station in source.target_stations]
    )
    trial_time = 2 * comm_distance / C
    eta = P_LINK * np.exp((-1) * comm_distance / L_ATT)
    num_trials = np.random.geometric(eta)
    time_taken = num_trials * trial_time
    return time_taken


world = World()

# creating stations
stations = []
for i in range(N):
    station = Station(
        world=world, position=np.array([np.cos(radiants[i]), np.sin(radiants[i])])
    )
    stations += [station]

# creating sourcea
source = MultiSchedulingSource(
    world=world,
    position=np.array([0, 0]),
    target_stations=stations,
    time_distribution=time_distribution,
    state_generation=state_generation,
)

protocol = VerifyProtocol()

protocol.setup(world=world, communication_speed=C)

current_message = None

# here protocol begins
# create angles
current_message = {"event": "send"}
current_message = protocol.check(current_message)
current_message = world.event_queue.resolve_next_event()

parity, angles_list = qaf.angles(N)
order = np.random.permutation(N)
meas_results = []
for i in range(N):
    base = [
        1 / np.sqrt(2) * (mat.z0 + np.exp(1j * angles_list[i]) * mat.z1),
        1 / np.sqrt(2) * (mat.z0 - np.exp(1j * angles_list[i]) * mat.z1),
    ]
    current_message["event"] = "measure"
    current_message["base"] = base
    current_message["actor"] = stations[i]
    protocol.check(current_message)
    current_message = world.event_queue.resolve_next_event()
    meas_results += [current_message["measurement_outcome"]]
print(np.sum(meas_results) % 2 == parity)
