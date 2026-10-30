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
from requsim.events import TCP_Purifying_Event_graph, MultiSourceEvent
from requsim.tools.evaluation import standard_graph_state_evaluation
import pandas as pd
import matplotlib.pyplot as plt
from requsim.tools.noise_channels import w_noise_channel, z_noise_channel


def construct_white_noise_channel(error_prob):
    def white_noise_func(rho):
        return w_noise_channel(rho=rho, alpha=error_prob)

    return NoiseChannel(n_qubits=1, channel_function=white_noise_func)


def construct_dephasing_noise_channel(dephasing_time):
    def lambda_dp(t):
        return (1 - np.exp(-t / dephasing_time)) / 2

    def dephasing_noise_func(rho, t):
        return z_noise_channel(rho=rho, epsilon=lambda_dp(t))

    return NoiseChannel(n_qubits=1, channel_function=dephasing_noise_func)


class TCP_graph_protocol_scenario_2(Protocol):

    def __init__(self, graph, world=None, communication_speed=None, max_distance=None):
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
        self.max_distance = max_distance
        self.graph = graph
        super().__init__(world=world)

    @property
    def data(self):
        return pd.DataFrame({"time": self.time_list, "state": self.state_list})

    def setup(self, world=None, communication_speed=None, max_distance=None):
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

        if max_distance is None:
            raise ValueError(
                "max_distance is not specified. "
                + "Must be provided as part of the setup method"
            )
        else:
            self.max_distance = max_distance

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

    def check(self, message=None, e=0):
        """check current status and schedule new events.

        looks globally at the status of the whole 'world' and decides which events need to be scheduled.
        """
        graph_groups = self._get_graph_state_groups()
        num_graph = len(graph_groups)
        num_graph_scheduled = len(self._graph_groups_scheduled())
        # if no scheduled event and no ghz pair sent then send
        if message is not None:
            try:
                if message["eval"] is True:
                    for g in graph_groups:
                        self._eval_graph_state(g)
                        for qbit in g.qubits:
                            qbit.destroy()
                        g.destroy()
            except KeyError:
                pass
        if e == 2:
            # do the P2 algorithm
            if message is not None:
                try:
                    if message["is_successful"] is True:
                        return "success_P2"
                    else:
                        return "fail_P2"
                except KeyError:
                    pass
            if num_graph == 2:
                tcp_event = TCP_Purifying_Event_graph(
                    time=self.world.event_queue.current_time,
                    multiqubits=graph_groups,
                    graph=self.graph,
                    communication_time=self.max_distance / self.communication_speed,
                    option="P2",
                )
                self.world.event_queue.add_event(tcp_event)
        else:
            if message is not None:
                try:
                    if message["is_successful"] is True:
                        return "success_P1"
                    else:
                        return
                except KeyError:
                    pass
            """
                try:
                    if message["unblocked_objects"] is not None:
                            for g in graph_groups:# assume there is only one graph state
                                self._eval_graph_state(g)
                                for qbit in g.qubits:
                                    qbit.destroy()
                            g.destroy()
                except KeyError:
                    #do nothing
                    pass
            """
            if (
                num_graph + num_graph_scheduled < 2 + e
            ):  # we need two copies for one TCP protocol execution
                self.source_central.schedule_event()

            # if both graph states are there, start tcp protocol
            if num_graph == 2 + e:
                tcp_event = TCP_Purifying_Event_graph(
                    time=self.world.event_queue.current_time,
                    multiqubits=graph_groups[e:],
                    graph=self.graph,
                    communication_time=self.max_distance / self.communication_speed,
                )
                self.world.event_queue.add_event(tcp_event)


def run(w_noise, mem_noise, max_iter, graph, params):
    C = params["COMMUNICATION_SPEED"]
    P_LINK = params["P_LINK"]
    LENGTH = params["LENGTH"]
    L_ATT = params["L_ATT"]

    def state_generation(source):
        state = graph.rho
        w_channel = construct_white_noise_channel(w_noise)
        for i in range(graph.N):
            state = w_channel.apply_to(rho=state, qubit_indices=[i])
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
            eta**graph.N
        )  # change here in terms of how many stations we have
        time_taken = num_trials * trial_time
        return time_taken

    world = World()
    # creating stations
    radiants = np.linspace(0, 2 * np.pi, graph.N + 1)
    stations = []
    for i in range(graph.N):
        station = Station(
            world=world,
            position=np.array(
                [LENGTH * np.cos(radiants[i]), LENGTH * np.sin(radiants[i])]
            ),
            memory_noise=construct_dephasing_noise_channel(mem_noise),
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
    protocol.setup(world=world, communication_speed=C, max_distance=LENGTH)

    current_message = None
    counter = 0
    evaluate = False
    while len(protocol.time_list) < max_iter:
        inter_message = protocol.check(message=current_message, e=counter)
        if inter_message is not None:
            if inter_message == "success_P1":
                counter += 1
            else:
                counter = 0
                # maybe make this part better (w/o continue)
                current_message = world.event_queue.resolve_next_event()
                current_message.update({"eval": True})
                continue

        current_message = world.event_queue.resolve_next_event()
    return protocol


if __name__ == "__main__":
    params = {
        "P_LINK": 0.80,
        "LENGTH": 20e3,
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
        sets=[[0], [1, 2, 3]],
    )
    white_noise_list = np.linspace(0.6, 1, num=10)
    mem_noise_list = np.logspace(-3, 10, num=10, base=2)
    max_iter = 10
    raw_data = [
        run(
            w_noise=w_noise,
            mem_noise=mem_noise,
            graph=graph,
            max_iter=max_iter,
            params=params,
        ).data
        for mem_noise in mem_noise_list
        for w_noise in white_noise_list
    ]
    results_list = [
        standard_graph_state_evaluation(data_frame=df, graph=graph, option="comp")
        for df in raw_data
    ]
    index = pd.MultiIndex.from_product(
        [mem_noise_list, white_noise_list], names=["mem_noise", "white_noise"]
    )

    results = pd.DataFrame(
        data=results_list,
        index=index,
        columns=[
            "raw_rate",
            "fidelity",
            "fidelity_std_err",
        ],
    )
    print(results)
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(white_noise_list, mem_noise_list)
    ax.pcolormesh(
        X, Y, results.values[:, 1].reshape(len(mem_noise_list), len(white_noise_list))
    )
    plt.savefig("../images/pcolormesh2_withP2.png")
