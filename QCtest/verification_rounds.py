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
from requsim.events import MeasurementEvent, MultiSourceEvent
from requsim.quantum_objects import Station, MultiSource
from requsim.world import World

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


class VerificationProtocol(Protocol):
    def __init__(self):
        self.time_list = []
        self.state_list = []
        super(VerificationProtocol, self).__init__(world=None)

    @property
    def data():
        return pd.Dataframe({"time": self.time_list, "state": self.state_list})

    def setup(self, world=None, communication_speed=None, rng=None):
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

        assert len(actors) >= 2
        # more station checks?
        self.actors = actors

        sources = self.world.world_objects["Source"]
        assert len(sources) == 1
        assert (sources[0].position == [0, 0]).all
        self.source = sources[0]
        # more checks?

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def _get_multiqubit(self, N):
        try:
            multiqubit = self.world.world_objects[f"{N}-qubit MultiQubit"]
        except KeyError:
            multiqubit = None
        return multiqubit[0]

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
            # multiqubit = self._get_multiqubit()
            # needs argument of size because of the label
            # maybe better check or change label?
            self.source.schedule_event()
        # measure GHZ (with bases)
        if message["event"] == "measure":
            current_N = len(self.world.world_objects["Qubit"])
            multiqubit = self._get_multiqubit(current_N)
            bases = message["bases"]

            # make sure actor and basis order is the same respectivley
            for i, base in enumerate(bases):
                measure_event = MeasurementEvent(
                    time=self.world.event_queue.current_time,
                    station=self.actors[i],
                    base=base,
                    rng=self.rng,
                )
                self.world.event_queue.add_event(measure_event)


if __name__ == "__main__":
    N = 3
    world = World()
    init_state = mat.ghz(N) @ mat.H(mat.ghz(N))
    stations = []
    for i in range(N):
        stations += [Station(world=world, position=[i, 0])]
    ghz_source = MultiSource(world=world, position=[0, 0], target_stations=stations)
    ghz_event = MultiSourceEvent(
        time=world.event_queue.current_time, source=ghz_source, initial_state=init_state
    )
    world.event_queue.add_event(ghz_event)
    world.event_queue.resolve_next_event()
    for j in range(N):
        meas_event = MeasurementEvent(
            time=world.event_queue.current_time,
            station=stations[j],
            base=[mat.z0, mat.z1],
        )
        world.event_queue.add_event(meas_event)
    world.print_status()
    while world.event_queue.next_event is not None:
        res = world.event_queue.resolve_next_event()
        print(res["measurement_outcome"])
        world.print_status()
