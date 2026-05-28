"""
Verification protocol using requsim
"""

from requsim.world import World
from requsim.quantum_objects import Station, SourceMult, MultiQubit
import requsim.libs.matrix as mat
import numpy as np
from requsim.events import MultiSourceEvent, MeasurementEvent
from requsim.libs.aux_functions import distance
from verification_rounds import angles

C = 2e8  # speed of light

N = 10  # number of users

speedMeas = 1e-9

# put stations around source in a circle
radius = 2e3

# N+1 since we dont want the first and last to overlap
radiants = np.linspace(0, 2 * np.pi, N + 1)

world = World()
# iterate later
stations = []
for i in range(N):
    station = Station(
        world=world, position=np.array([np.cos(radiants[i]), np.sin(radiants[i])])
    )
    stations += [station]
source = SourceMult(
    world=world,
    position=np.array([0, 0]),
    target_stations=stations,
)

# density matrix
ghz_rho = mat.ghz(N) @ mat.H(mat.ghz(N))

# delays are the same here
delay = distance(source, stations[0]) / C

event_source = MultiSourceEvent(
    time=world.event_queue.current_time + delay,
    source=source,
    initial_state=ghz_rho,
)

world.event_queue.add_event(event_source)

result = world.event_queue.resolve_next_event()

# verifier calculates angles

angles_list = angles(N)

meas_results = []
for i in range(N):
    base = [
        1 / np.sqrt(2) * (mat.z0 + np.exp(1j * angles_list[i]) * mat.z1),
        1 / np.sqrt(2) * (mat.z0 - np.exp(1j * angles_list[i]) * mat.z1),
    ]
    event_measure = MeasurementEvent(
        time=world.event_queue.current_time + speedMeas,
        multiqubit=result["output_state"],
        station=stations[i],
    )
    world.event_queue.add_event(event_measure)
    result = world.event_queue.resolve_next_event()
    meas_results += [result["measurement_outcome"]]


print(meas_results)
