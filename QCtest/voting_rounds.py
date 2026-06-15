"""
The Voting protocol class which takes over all the interactions with a Quantum Network via simulation with requsim
"""

from requsim.tools.protocol import Protocol
from requsim.libs import matrix as mat
from requsim.events import MeasurementEvent


class VotingProtocol(Protocol):
    def __init__(self):
        self.time_list = []
        self.state_list = []
        super(VotingProtocol, self).__init__(world=None)

    @property
    def data():
        return pd.Dataframe({"time": self.time_list, "state": self.state_list})

    def setup(self, world=None, communication_speed=None, rng=None):
        """
        Should be run after the relevant WorldObjects have been added to the world.

        Parameters
        ----------
        world: World
            The World object representing the scenario for which this Protocol will be used.
        communication_speed: scalar
            The communication speed used for calculatinf the delays when sending qubits or classical messages between stations.

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

        self.rng = rng

        actors = self.world.world_objects["Station"]

        assert len(actors) >= 2

        self.actors = actors

        sources = self.world.world_objects["Source"]

        assert len(sources) == 1

        # do we even want this check?
        assert (sources[0].position == [0, 0]).all

        self.source = sources[0]
        # more check

    def _get_multiqubit(self, N):
        try:
            multiqubit = self.world.world_objects[f"{N}-qubit Multiqubit"]
        except KeyError:
            multiqubit = None
            return None
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
        # measure GHZ (in X basis)
        if message["event"] == "measure":
            current_N = len(self.world.world_objects["Qubit"])
            multiqubit = self._get_multiqubit(current_N)
            base = [mat.x0, mat.x1]
            for i, actor in enumerate(self.actors):
                measure_event = MeasurementEvent(
                    time=self.world.event_queue.current_time,
                    station=actor,
                    base=base,
                    rng=self.rng,
                )
                self.world.event_queue.add_event(measure_event)
