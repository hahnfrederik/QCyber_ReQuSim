"""
Verification protocol using requsim
"""

from requsim.world import World
from requsim.quantum_objects import (
    Station,
    MultiSource,
    MultiQubit,
    SchedulingSource,
    MultiSchedulingSource,
)
import requsim.libs.matrix as mat
import numpy as np
from requsim.events import MultiSourceEvent, MeasurementEvent
from requsim.libs.aux_functions import distance
from verification_rounds import VerifyProtocol
from voting_rounds import VotingProtocol
import QC_aux_functions as qaf

C = 2e8  # speed of light

N = 10  # number of users
C = 3
P_LINK = 0.80
T_DP = 100e-3
LAMBDA_BSM = 0.99
L_ATT = 22e3
rng = np.random.default_rng()
votes = rng.choice(C, size=N)
K = int(np.ceil(np.log2(C)))
P = 2
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

# creating sources
source = MultiSchedulingSource(
    world=world,
    position=np.array([0, 0]),
    target_stations=stations,
    time_distribution=time_distribution,
    state_generation=state_generation,
)

# protocol = VerifyProtocol()

# protocol.setup(world=world, communication_speed=C)

# current_message = None

## here protocol begins
## create angles
# current_message = {"event": "send"}
# current_message = protocol.check(current_message)
# current_message = world.event_queue.resolve_next_event()

# parity, angles_list = qaf.angles(N)
# order = np.random.permutation(N)
# meas_results = []
# bases= []
# for i in range(N):
#    bases += [[
#        1 / np.sqrt(2) * (mat.z0 + np.exp(1j * angles_list[i]) * mat.z1),
#        1 / np.sqrt(2) * (mat.z0 - np.exp(1j * angles_list[i]) * mat.z1),
#    ]]
# current_message["event"] = "measure"
# current_message["bases"] = bases
# current_message["actor"] = stations[i]
# protocol.check(current_message)
# while world.event_queue.next_event is not None:
#    current_message = world.event_queue.resolve_next_event()
#    meas_results += [current_message["measurement_outcome"]]
# print(np.sum(meas_results) % 2 == parity)

# from here we do the voting subroutine
tally = np.empty((K, P, N, N))
# the input of the plurality votes of the users
votes_bin = [np.binary_repr(vote, width=K) for vote in votes]

# secret ordering
uniq_ordering = rng.permutation(N)

# TODO: check if veriifying or voting

# TODO: if verifying then verify

# If voting

# TODO: calculate ratio and check

# TODO: check current subround
# for k in range(K):
#    for p in range(P):
#        for n in range(N):

vot_protocol = VotingProtocol()

vot_protocol.setup(world=world, communication_speed=C, rng=rng)

print(votes)
print(uniq_ordering)
for k in range(K):
    rand_bits = np.zeros(N)
    for p in range(P):
        for n in range(N):
            agent = uniq_ordering[n]
            vot_protocol.check(message={"event": "send"})
            world.event_queue.resolve_next_event()

            vot_protocol.check(message={"event": "measure"})
            meas_res = []
            while world.event_queue.next_event is not None:
                res = world.event_queue.resolve_next_event()
                meas_res += [res["measurement_outcome"]]
            # print(meas_res)
            for j in range(N):
                if j == agent:
                    if p == P - 1:
                        # we also need to check that it all fits
                        if rand_bits[j] == int(votes_bin[agent][k]):
                            inp = (0 + meas_res[j]) % 2
                        else:
                            inp = (1 + meas_res[j]) % 2
                    else:
                        r_a = rng.choice(2)
                        inp = (meas_res[j] + r_a) % 2
                        rand_bits[j] += r_a
                else:
                    inp = meas_res[j]
                tally[k][p][n][j] = inp
# print(tally)
outcome_votes_bin = np.sum(np.sum(tally, axis=3), axis=1) % 2
outcome_votes_bin = np.transpose(outcome_votes_bin, (1, 0))
print(outcome_votes_bin)
# print(np.sum(np.sum(tally[k], axis = 2), axis=0)%2)
outcome_votes = []
for binl in outcome_votes_bin:
    h = ""
    for bit in binl:
        h += str(int(bit))
    outcome_votes += [int(h, 2)]
print(outcome_votes)
print("outcome votes statistic", np.unique(outcome_votes, return_counts=True))
print("input votes statistic", np.unique(votes, return_counts=True))
