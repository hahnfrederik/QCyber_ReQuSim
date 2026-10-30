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
from requsim.tools.protocol import Protocol
from requsim.noise import NoiseChannel, NoiseModel
import pandas as pd
from requsim.tools.evaluation import standard_ghz_evaluation
from requsim.events import ConnectBellsToGHZEvent, MultiSourceEvent
from warnings import warn


# two different TCP implementations. One based on graph state basis and on ein the computational basis

# for graph state basis, we switch from computational to graph state basis and then we use the tcp protocol on it
# all the errors that we want to put on the state should be also translated to the graph sate basis.


class TCP_graph_protocol_scenario_3(Protocol):

    def __init__(self, N, world=None, communication_speed=None):
        if world is not None or communication_speed is not None:
            warn(
                "Initializing Protocol with setup-dependent arguments (like world) is no longer recommended "
                + "and may be deprecated in future versions. "
                + "Protocols should be initializeable without tying it to a specific scenario. "
                + "Use the setup method to pass scenario-dependent arguments instead.",
                FutureWarning,
                stacklevel=2,
            )
        self.time_list = []
        self.N = N  # number of outlying stations
        self.state_list = []
        self.communication_speed = communication_speed
        super().__init__(world=world)

    @property
    def data(self):
        return pd.DataFrame({"time": self.time_list, "state": self.state_list})

    def setup(self, world=None, communication_speed=None):
        """
        create the sources and so on

        """
        if world is None:
            if self.world is None:
                raise ValueError(
                    "world is not specified. "
                    + "Must be provided either as part of the initialization (deprecated) or "
                    + "the setup methos (recommended)."
                )
            else:
                pass
        else:
            self.world = world
        if communication_speed is None:
            if self.communication_speed is None:
                raise ValueError(
                    "communication_speed is not specified. "
                    + "Must be provided either as part of the initialization (deprecated) or"
                    + "the setup method (recommended)."
                )
            else:
                pass
        else:
            self.communication_speed = communication_speed

        stations = self.world.world_objects["Station"]
        assert len(stations) == self.N + 1
        for i in range(len(stations)):
            if isinstance(stations[i].position, int):
                if not (station[i].position.shape[0] == 2):
                    raise ValueError("position of stations should be in 2 dimensions")
        self.stations = stations
        central_stations = list(
            filter(lambda x: (x.position == np.array([0, 0])).all(), self.stations)
        )
        assert len(central_stations) == 1
        self.station_central = central_stations[0]

        sources = self.world.world_objects["Source"]
        assert len(sources) == self.N
        self.sources_central = sources

        for source in self.sources_central:
            assert callable(getattr(source, "schedule_event", None))

    def _get_pairs(self, station):
        try:
            pairs = self.world.world_objects[f"Pair"]
        except KeyError:
            pairs = []
        return list(
            filter(
                lambda x: x.is_between_stations(station, self.station_central),
                pairs,
            )
        )

    def _get_multiqubits(self):
        try:
            multiqubits = self.world.world_objects[f"{self.N}-qubit MultiQubit"]
        except KeyError:
            multiqubits = []
        return multiqubits

    def _pairs_scheduled(self, station):
        return list(
            filter(
                lambda event: (
                    isinstance(event, MultiSourceEvent)
                    and (station in event.source.target_stations)
                    and (self.station_central in event.source.target_stations)
                ),
                self.world.event_queue.queue,
            )
        )

    def _eval_ghz_state(self, g):
        dists = []
        for i in range(self.N):
            dists += [distance(self.station_central, self.stations[i])]
            comm_distance = np.max(dists)
            comm_time = comm_distance / self.communication_speed

            self.time_list += [self.world.event_queue.current_time + comm_time]
            self.state_list += [g.state]
            return

    def check(self, message=None):
        """check current status and schedule new events.

        looks globally at the status of the whole 'world' and decides which events need to be scheduled.
        """
        num_pairs = []
        num_pairs_scheduled = []
        pairs = []
        multiqubits = self._get_multiqubits()
        for station in self.stations[:-1]:
            pair = self._get_pairs(station)
            num_pairs += [len(pair)]
            num_pairs_scheduled += [len(self._pairs_scheduled(station))]
            pairs += [pair]

        # for every station, if no scheduled pair, then send
        for i in range(len(self.stations[:-1])):
            if num_pairs[i] + num_pairs_scheduled[i] == 0:
                self.sources_central[i].schedule_event()

        # if all stations have a qubit, then combine to GHZ state
        if (np.array(num_pairs) >= 1).all():
            self.world.print_status()
            make_ghz_event = ConnectBellsToGHZEvent(
                time=self.world.event_queue.current_time,
                pairs=[list_pair[0] for list_pair in pairs],
                station=self.stations[-1],
            )
            self.world.event_queue.add_event(make_ghz_event)
        if multiqubits:
            for mqubit in multiqubits:
                self._eval_ghz_state(mqubit)
                for qubit in mqubit.qubits:
                    qubit.destroy()
                mqubit.destroy()


def run(length, max_iter, N, params):
    C = params["COMMUNICATION_SPEED"]
    P_LINK = params["P_LINK"]
    T_DP = params["T_DP"]
    LAMBDS_MEAS = params["LAMBDA_MEAS"]
    L_ATT = params["L_ATT"]

    def state_generation(source):
        state = mat.phiplus @ mat.H(mat.phiplus)
        comm_distance = max(
            [
                distance(source, source.target_stations[i])
                for i in range(len(source.target_stations))
            ]
        )
        storage_time = 2 * comm_distance / C
        for idx, station in enumerate(source.target_stations):
            if station.memory_noise is not None:
                state = station.memory_noise.apply_to(
                    rho=state, qubit_indices=[idx], t=storage_time
                )
        return state

    def time_distribution(source):
        comm_distance = max(
            [
                distance(source, source.target_stations[i])
                for i in range(len(source.target_stations))
            ]
        )
        trial_time = 2 * comm_distance / C
        eta = P_LINK * np.exp(-comm_distance / L_ATT)
        num_trials = np.random.geometric(
            eta
        )  # change here in terms of how many stations we have
        time_taken = num_trials * trial_time
        return time_taken

    def Meas_error_func(rho):
        return LAMBDA_MEAS * rho + (1 - LAMBDA_MEAS) * mat.I(graph.N) / graph.N

    Meas_noise_channel = NoiseChannel(n_qubits=2, channel_function=Meas_error_func)
    Meas_error_model = NoiseModel(channel_before=Meas_noise_channel)

    world = World()
    # creating stations

    radiants = np.linspace(0, 2 * np.pi, N + 1)
    stations = []
    for i in range(N):
        station = Station(
            world=world,
            position=np.array(
                [length * np.cos(radiants[i]), length * np.sin(radiants[i])]
            ),
        )
        stations += [station]

    station_central = Station(
        world=world,
        position=np.array([0, 0]),
        memory_noise=None,  # for now
        BSM_noise_model=Meas_error_model,
    )
    stations += [station_central]

    sources = []
    for i in range(N):
        source = MultiSchedulingSource(
            world=world,
            position=np.array([0, 0]),
            target_stations=[station_central, stations[i]],
            time_distribution=time_distribution,
            state_generation=state_generation,
        )
        sources += [source]

    protocol = TCP_graph_protocol_scenario_3(N)
    protocol.setup(world=world, communication_speed=C)

    current_message = None
    while len(protocol.time_list) < max_iter:
        # world.print_status()
        protocol.check(message=current_message)
        current_message = world.event_queue.resolve_next_event()

    return protocol


if __name__ == "__main__":
    params = {
        "P_LINK": 0.80,
        "T_DP": 100e-3,
        "LAMBDA_MEAS": 0.99,
        "COMMUNICATION_SPEED": 2e8,
        "L_ATT": 22e3,
    }

    N = 4
    length_list = np.linspace(1, 10, num=8)
    max_iter = 3
    raw_data = [
        run(length=length, N=N, max_iter=max_iter, params=params).data
        for length in length_list
    ]
    result_list = [standard_ghz_evaluation(data_frame=df) for df in raw_data]
    results = pd.DataFrame(
        data=result_list,
        index=length_list,
        columns=[
            "raw_rate",
            "fidelity",
            "fidelity_std_err",
        ],
    )
    print(results)
