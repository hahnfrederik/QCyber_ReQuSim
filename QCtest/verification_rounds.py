"""
Verification protocol form paper Experimental Quantum Electronic Voting

"""

import sys
import os
import numpy as np
import requsim.libs.matrix as mat
import requsim.libs.aux_functions as af
import requsim.tools.noise_channels as nc
from requsim.tools.protocol import Protocol

# function for ghz fidelity

C = 2e8  # speed of light
speedMeas = 1e-9  # speed of one photonic measurement


def ghz_fidelity(rho: np.ndarray, num_parties):
    z0s = [mat.z0] * num_parties
    z0s = mat.tensor(*z0s)
    z1s = [mat.z1] * num_parties
    z1s = mat.tensor(*z1s)
    ghz_psi = 1 / np.sqrt(2) * (z0s + z1s)

    fidelity = np.real_if_close(np.dot(np.dot(mat.H(ghz_psi), rho), ghz_psi)[0, 0])

    return fidelity


# create quantum state that is epsilon far from ghz state as per paper
def eps_error(rho, N, noise_choice=0, index=0, error=0):
    if noise_choice == 0:
        rho = af.apply_single_qubit_map(nc._x_noise_function, index, rho, error)
        noise = "x_noise"
    elif noise_choice == 1:
        rho = af.apply_single_qubit_map(nc._y_noise_function, index, rho, error)
        noise = "y_noise"
    elif noise_choice == 2:
        rho = af.apply_single_qubit_map(nc._z_noise_function, index, rho, error)
        noise = "z_noise"
    elif noise_choice == 3:
        # for w_noise we do one minus, since a the error probability is reversed in comparison to others
        rho = af.apply_single_qubit_map(
            _w_noise_function, index, rho, 1 - error
        )  # Maybe for consistency we should rewrite the _w_noise_function
        noise = "w_noise"
    else:
        rho = af.apply_single_qubit_map(nc._ad_noise_function, index, rho, error)
        noise = "ad_noise"
    # print(epsilon)
    return rho, noise


class VerifyProtocol(Protocol):
    def __init__(self):
        self.time_list = []
        self.state_list = []
        super(Verify, self).__init__(world=None)

    @property
    def data():
        return pd.Dataframe({"time": self.time_list, "state": self.state_list})

    def setup(self, world=None, communitcation_speed=None):
        """
        Should be run after the relevant WorldObjects have been added
        to the world.

        Parameters
        ----------
        world : World
            The World object representing the scenario for which this Protocol will be used.
        communication_speed : scalar
            The communication speed usd for calculating delays when sending qubits or
            classical messages between stations.


        Returns
        -------
        None

        """
        if world is None:
            if self.world is None:
                raise ValueError(
                    "world is not specified. "
                    + "Must be provided either as part of the initialization (deprecated) or "
                    "the setup method (recommended)."
                )
            else:
                pass
        else:
            self.world = world

        if communication_speed is None:
            if self.communication_speed is None:
                raise ValueError(
                    "communication_speed is not specified. "
                    + "Must be provided either as part of the initialization (deprecated) or "
                    + "the setup method (recommended)."
                )
            else:
                pass
        else:
            self.communication_speed = communication_speed

        actors = self.world.world_objects["Station"]

        assert len(stations) >= 2
        # more station checks?
        self.stations = stations

        sources = self.world.world_objects["Source"]
        assert len(sources) == 1
        assert sources[0].position == (0, 0)
        self.source = sources[0]
        # more checks?

    def _get_multiqubit(self):
        try:
            multiqubit = self.world.world_objects["MultiQubit"]
        except KeyError:
            multiqubit = None
        return multiqubit

    def _get_multiqubit_scheduled(self):
        return list(
            filter(
                lambda event: (isinstance(event, MultiSourceEvent)),
                self.world.event_queue.queue,
            )
        )

    def check(self, message=None):
        """
        checks status of world and does events and calculations based upon that
        """

        # 2 different scanrios
        # send GHZ
        if message["event"] == "send":
            multiqubit = self._get_multiqubit()
            if multiqubit == None:
                self.source.schedule_event
        # measure GHZ (with bases)
        if message["event"] == "measure":
            multiqubit = self._get_multiqubit()
            base = message["base"]
            actor = message["actor"]
            measure_event = MeasurementEvent(
                time=world.event_queue.current_time,
                multiqubit=multiqubit,
                station=actor,
                base=base,
            )
            self.event_queue.add_event(measure_event)


def angles(N):
    angles = np.random.rand(N - 1) * np.pi
    # pick parity m uniformly from {0, 1} and then make sure the sum is m*pi
    m = np.random.randint(0, 2)  # or any integer really, but parity is what matters
    last_angle = (
        m * np.pi - np.sum(angles)
    ) % np.pi  # @Jan: I think here we can actually just do -np.sum(angles) % np.pi
    angles = np.append(angles, last_angle)
    return angles


def verify(world, rho, verifier=None):
    # not using verifier now, but inserted for later
    N = int(np.log2(rho.shape[0]))

    protocol = VerifyProtocol()
    angles_list = angles(N)

    results = []
    for i in range(N):
        protocol.check({"message": "send"})
        state_vec_1 = 1 / np.sqrt(2) * (mat.z0 + np.exp(1j * angles_list[i]) * mat.z1)
        state_vec_2 = 1 / np.sqrt(2) * (mat.z0 - np.exp(1j * angles_list[i]) * mat.z1)

        proj = []
        # compute projectors
        if i < N - 1:
            proj.append(
                mat.tensor(state_vec_1 @ mat.H(state_vec_1), mat.I(2 ** (N - 1 - i)))
            )
            proj.append(
                mat.tensor(state_vec_2 @ mat.H(state_vec_2), mat.I(2 ** (N - 1 - i)))
            )
        else:
            proj.append(state_vec_1 @ mat.H(state_vec_1))
            proj.append(state_vec_2 @ mat.H(state_vec_2))

        # calculate probabilites
        probs = []
        # print(rho.shape)
        # print(i)
        p1 = np.trace(proj[0] @ rho)
        p2 = np.trace(proj[1] @ rho)
        if np.isclose(np.real(p1), 0):
            p1 = 0 + np.imag(p1) * 1j
        if np.isclose(np.real(p2), 0):

            p2 = 0 + np.imag(p2) * 1j
        p1 = np.real_if_close(p1)
        p2 = np.real_if_close(p2)
        assert np.imag(p1) == 0, p1
        assert np.imag(p2) == 0, p2
        assert np.real(p1) >= 0, p1
        assert np.real(p2) >= 0, p2
        probs.append(np.real(p1))
        probs.append(np.real(p2))

        # sanity check: probs[0] + probs[1] should be 1
        choice = np.random.choice(2, 1, p=[probs[0], probs[1]])[0]
        results.append(choice)

        # some sanity checks for rho_new
        if i != N - 1:
            # collapse of state
            rho_new = proj[choice] @ rho @ proj[choice] / probs[choice]
            rho = mat.ptrace(rho_new, [0])

    ### I think it is safer to make sure that both sides are integers so maybe:
    expected_parity = int(round(np.sum(angles_list) / np.pi)) % 2
    measured_parity = int(np.sum(results) % 2)
    return expected_parity == measured_parity


if __name__ == "__main__":
    N = 4
    params = {
        "N": 4,
        "P_LINK": 0.80,
        "T_DP": 100e-3,
        "LAMBDA_BSM": 0.99,
        "COMMUNICATION_SPEED": 2e8,
    }

    # create ghz state
    rho = mat.ghz(N) @ mat.H(mat.ghz(N))
    # create state that is epsilon far from ghz

    min_eps = 0.1

    t_eps = None

    for noise_choice in range(5):
        # exponent for error parameter
        exp = -1
        # better name? variable to get a more fine tuned exponent
        step = 0
        while t_eps is None or not np.isclose(t_eps, min_eps):
            rho2, noise_string = eps_error(
                rho, N, noise_choice, index=0, error=10**exp
            )
            t_eps = np.sqrt(1 - (ghz_fidelity(rho2, N)))

            if t_eps > min_eps:
                exp -= 1 * (10**step)
            if t_eps < min_eps:
                exp += 1 * (10**step)
                step -= 1
                exp -= 1 * (10**step)

        # make iterN dependent on the probability
        p = (t_eps**2) / 4
        iterN = int(15 / p)
        print("-----------------------------------")
        print("the noise being used is", noise_string)
        print("the state being used is", t_eps, "far from the ghz state")
        print(
            "theoretical minimum probability of state failing verification protocol:", p
        )

        hits = 0
        print("using", iterN, "iterations in simulation")
        for i in range(iterN):
            hits += verify(rho2)
        print(
            "simulation probability of the state failing verification:",
            1 - hits / iterN,
        )
        t_eps = None
