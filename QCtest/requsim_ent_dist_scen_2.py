import graphepp as gg
from GraphsReq import GraphsReq
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
from requsim.events import TCP_Purifying_Event_graph
from requsim.tools.evaluation import standard_graph_state_evaluation
import pandas as pd


class TCP_graph_protocol_scenario_2(Protocol):

    def __init__(self, graph, world=None, communication_speed=None):
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
        self.state_list = []
        self.communication_speed = communication_speed
        self.graph = graph
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
        assert len(stations) == self.graph.N
        if isinstance(stations[0].position, int):
            raise ValueError("position of stations should be in 2 dimensions")
        else:
            self.stations = stations

        sources = self.world.world_objects["Source"]
        assert len(sources) == 1
        self.source_central = sources[0]

        assert callable(getattr(self.source_central, "schedule_event", None))

        # number of schedules pair needed?

    def _get_graph_state_groups(self):
        try:
            graph_states = self.world.world_objects[f"{self.graph.N}-qubit MultiQubit"]
        except KeyError:
            graph_states = []
        return graph_states

    def _graph_groups_scheduled(self):
        return list(
            filter(
                lambda event: (isinstance(event, MultiSourceEvent)),
                self.world.event_queue.queue,
            )
        )

    def _eval_graph_state(self, g):
        dists = []
        for i in range(self.graph.N):
            dists += [distance(self.source_central, self.stations[i])]
            comm_distance = np.max(dists)
            comm_time = comm_distance / self.communication_speed

            self.time_list += [self.world.event_queue.current_time + comm_time]
            self.state_list += [g.state]
            return

    def check(self, message=None):
        """check current status and schedule new events.

        looks globally at the status of the whole 'world' and decides which events need to be scheduled.
        """
        graph_groups = self._get_graph_state_groups()
        num_graph = len(graph_groups)
        num_graph_scheduled = len(self._graph_groups_scheduled())
        # if no scheduled event and no ghz pair sent then send
        if (
            num_graph + num_graph_scheduled < 2
        ):  # we need two copies for one TCP protocol execution
            self.source_central.schedule_event()

        # if both graph states are there, start tcp protocol
        if num_graph == 2:
            tcp_event = TCP_Purifying_Event_graph(
                time=self.world.event_queue.current_time,
                multiqubits=graph_groups,
                graph=self.graph,
            )
            self.world.event_queue.add_event(tcp_event)

            for g in graph_groups:
                self._eval_graph_state(g)
                for qubit in g.qubits:
                    qubit.destroy()
                g.destroy()


# for computational basis,...


def run(length, max_iter, graph, params):
    C = params["COMMUNICATION_SPEED"]
    P_LINK = params["P_LINK"]
    T_DP = params["T_DP"]
    LAMBDS_MEAS = params["LAMBDA_MEAS"]
    L_ATT = params["L_ATT"]

    def state_generation(source):
        state = graph.rho
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
    radiants = np.linspace(0, 2 * np.pi, graph.N + 1)
    stations = []
    for i in range(graph.N):
        station = Station(
            world=world,
            position=np.array([np.cos(radiants[i]), np.sin(radiants[i])]),
        )
        stations += [station]

    source_graph = MultiSchedulingSource(
        world=world,
        position=np.array([0, 0]),
        target_stations=stations,
        time_distribution=time_distribution,
        state_generation=state_generation,
    )

    protocol = TCP_graph_protocol_scenario_2(graph)
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

    graph = GraphsReq(
        N=4,
        E=[
            (0, 1),
            (0, 2),
            (0, 3),
        ],
    )
    length_list = np.linspace(20e3, 200e3, num=8)
    max_iter = 10
    raw_data = [
        run(length=length, graph=graph, max_iter=max_iter, params=params).data
        for length in length_list
    ]
    results_list = [
        standard_graph_state_evaluation(data_frame=df, graph=graph) for df in raw_data
    ]

    results = pd.DataFrame(
        data=results_list,
        index=length_list,
        columns=[
            "raw_rate",
            "fidelity",
            "fidelity_std_err",
        ],
    )
    print(results)
