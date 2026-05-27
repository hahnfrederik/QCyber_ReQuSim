from requsim.world import World
from requsim.quantum_objects import Station, SourceMult, MultiQubit
import requsim.libs.matrix as mat
import numpy as np
from requsim.events import MultiSourceEvent, MeasurementEvent
from requsim.libs.aux_functions import distance

C = 2e8  # speed of light

N = 3  # number of users

speedMeas = 1e-9

world = World()
# iterate later
station_a = Station(world=world, position=np.array([0, 2000]))
station_b = Station(world=world, position=np.array([-1000, -1000]))
station_c = Station(world=world, position=np.array([-1000, -1000]))
source = SourceMult(
    world=world,
    position=np.array([0, 0]),
    target_stations=[station_a, station_b, station_c],
)

# density matrix
ghz_rho = mat.ghz(N) @ mat.H(mat.ghz(N))

# delays
delay_a = distance(source, station_a) / C
delay_b = distance(source, station_b) / C
delay_c = distance(source, station_c) / C

event_source = MultiSourceEvent(
    time=world.event_queue.current_time + np.max([delay_a, delay_b, delay_c]),
    source=source,
    initial_state=ghz_rho,
)

world.event_queue.add_event(event_source)

# event_measure = MeasurementEvent(time=world.event_queue.current_time + speedMeas,  )

# not sure what delay to use, right now using maximum, with the logic, that at that time every station has a qubit from the shared resource
# source.generate_multi_qubit(initial_state=ghz_rho, time=np.max([delay_a, delay_b, delay_c]), source = source)

result = world.event_queue.resolve_next_event()

event_measure = MeasurementEvent(
    time=world.event_queue.current_time + speedMeas,
    multiqubit=result["output_state"],
    station=station_b,
)

world.event_queue.add_event(event_measure)

result = world.event_queue.resolve_next_event()

print(result)
