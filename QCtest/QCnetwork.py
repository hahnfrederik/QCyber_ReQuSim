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
from verification_rounds import VerificationProtocol
from voting_rounds import VotingProtocol
import QC_aux_functions as qaf
from requsim.tools.noise_channels import w_noise_channel, z_noise_channel
from requsim.noise import NoiseChannel

C = 2e8  # speed of light

N = 5  # number of users
Candidates = 2
M = 13
P_LINK = 0.80
T_DP = 100e-3
LAMBDA_BSM = 0.99
L_ATT = 22e3
LENGTH = 1
delta = 1
rng = np.random.default_rng()
votes = rng.choice(Candidates, size=N)
K = int(np.ceil(np.log2(Candidates)))
P = 2
speedMeas = 1e-9
initial_noise = 0

# put stations around source in a circle
radius = 2e3

# N+1 since we dont want the first and last to overlap
radiants = np.linspace(0, 2 * np.pi, N + 1)


def construct_white_noise_channel(error_prob):
    def white_noise_func(rho):
        return w_noise_channel(rho=rho, alpha=error_prob)

    return NoiseChannel(n_qubits=1, channel_function=white_noise_func)


def state_generation(source):
    ghz_state = mat.ghz(N) @ mat.H(mat.ghz(N))
    w_channel = construct_white_noise_channel(initial_noise)
    for i in range(N):
        ghz_state = w_channel.apply_to(rho=ghz_state, qubit_indices=[i])
    return ghz_state


def time_distribution(source):
    comm_distance = max(
        [distance(source, t_station) for t_station in source.target_stations]
    )
    trial_time = 2 * comm_distance / C
    eta = P_LINK * np.exp((-1) * comm_distance / L_ATT)
    num_trials = np.random.geometric(eta**N)
    time_taken = num_trials * trial_time
    return time_taken


world = World()

# creating stations
stations = []
for i in range(N):
    station = Station(
        world=world,
        position=np.array([LENGTH * np.cos(radiants[i]), LENGTH * np.sin(radiants[i])]),
    )
    stations += [station]

# creating source(s)
source = MultiSchedulingSource(
    world=world,
    position=np.array([0, 0]),
    target_stations=stations,
    time_distribution=time_distribution,
    state_generation=state_generation,
)

## here protocol begins

tally = np.empty((K, P, N, N))
# the input of the plurality votes of the users
votes_bin = [np.binary_repr(vote, width=K) for vote in votes]

# secret ordering
uniq_ordering = rng.permutation(N)

# creating protocols for voting and verifying
vot_protocol = VotingProtocol()
vot_protocol.setup(world=world, communication_speed=C, rng=rng)

ver_protocol = VerificationProtocol()
ver_protocol.setup(world=world, communication_speed=C, rng=rng)

# output to check later
print(votes)
print(uniq_ordering)
all_ratios = []
for k in range(K):
    rand_bits = np.zeros(N)
    for p in range(P):
        for n in range(N):
            agent = uniq_ordering[n]
            voted = False
            ratios = np.zeros((2, N))
            while not voted:
                coin = rng.choice(2, 1, p=[1 - 2 ** (-M), 2 ** (-M)])
                if coin == 0:
                    # verify
                    ver_agent = rng.choice(N, 1)[0]
                    ratios[0][ver_agent] += 1
                    current_message = {"event": "send"}
                    current_message = ver_protocol.check(current_message)
                    current_message = world.event_queue.resolve_next_event()

                    parity, angles_list = qaf.angles(N)
                    order = np.random.permutation(N)
                    meas_results = []
                    bases = []
                    for i in range(N):
                        bases += [
                            [
                                1
                                / np.sqrt(2)
                                * (mat.z0 + np.exp(1j * angles_list[i]) * mat.z1),
                                1
                                / np.sqrt(2)
                                * (mat.z0 - np.exp(1j * angles_list[i]) * mat.z1),
                            ]
                        ]
                    current_message["event"] = "measure"
                    current_message["bases"] = bases
                    current_message["actor"] = stations[i]
                    ver_protocol.check(current_message)
                    while world.event_queue.next_event is not None:
                        current_message = world.event_queue.resolve_next_event()
                        meas_results += [current_message["measurement_outcome"]]
                    # print(np.sum(meas_results) % 2 == parity)
                    if not np.sum(meas_results) % 2 == parity:
                        ratios[1][ver_agent] += 1
                else:
                    # vote
                    # ignore the ratios where ratio[0] is 0 (no verification has happend there yet)
                    ratio_ind = np.argwhere(ratios[0] > 0)
                    if (ratios[1, ratio_ind] / ratios[0, ratio_ind] > delta).any():
                        # abort
                        raise RuntimeError(
                            "in round ",
                            k,
                            p,
                            n,
                            " the state could not be verified",
                            ratios,
                            ratios[1] / ratios[0],
                        )
                    voted = True
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
            all_ratios += [ratios]
# TODO: verification of the tally for all actors
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

# from here we want to make a graph
import matplotlib.pyplot as plt

# print(len(all_ratios))
# print(all_ratios[0].shape)
rates = np.array([r[1] / r[0] for r in all_ratios]).reshape(P, N, N)

print(rates)
bg = "#2e3440"
colors = ["#d4a5ff", "#ffa43a"][:P]
alphas = np.linspace(0.45, 1.0, N)

fig, ax = plt.subplots(figsize=(13, 5.5))

# bars

group_width = 0.8
w = group_width / (P * N)
offsets = (np.arange(P * N) - (P * N - 1) / 2) * w

for p in range(P):
    for n in range(N):
        x = np.arange(N) + offsets[p * N + n]
        ax.bar(x, rates[p, n], width=w, color=colors[p], alpha=alphas[n], linewidth=0.3)

# ax.axhline(delta, color="#c58af9", linestyle="--", linewidth=2)

ax.set_xticks(np.arange(N))
ax.set_xticklabels([f"Voter {i+1}" for i in range(N)])
ax.set_xlim(-0.55, N - 0.45)
ax.set_ylim(0, np.max(rates))
ax.set_xlabel("Voters", fontsize=13)
ax.set_ylabel("Failure rate", fontsize=13)
ax.grid(axis="both", linestyle="--", alpha=0.15)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("../images/test.png")
