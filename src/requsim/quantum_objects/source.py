from . import WorldObject, Pair, mult_qubit
from .. import events


class Source(WorldObject):
    """A source of entangled pairs.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The two stations the source to which the source sends the entangled
        pairs, usually the neighboring repeater stations.
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The two stations the source to which the source sends the entangled
        pairs, usually the neighboring repeater stations.
    type : str
        "Source"

    """

    def __init__(self, world, position, target_stations, label=None):
        self.position = position
        self.target_stations = target_stations
        super(Source, self).__init__(world=world, label=label)

    def __str__(self):
        return (
            f"{self.label} generating states between stations "
            + ", ".join([x.label for x in self.target_stations])
            + "."
        )

    @property
    def type(self):
        return "Source"

    def generate_pair(self, initial_state):
        """Generate an entangled pair.

        The Pair will be generated in the `initial_state` at the
        `self.target_stations` of the source.
        Usually called from a SourceEvent.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial density matrix of the two-qubit

        Returns
        -------
        Pair
            The newly generated Pair.

        """
        station1 = self.target_stations[0]
        station2 = self.target_stations[1]
        qubit1 = station1.create_qubit()
        qubit2 = station2.create_qubit()
        return Pair(
            world=self.world, qubits=[qubit1, qubit2], initial_state=initial_state
        )


class SourceMult(Source):
    """A source of multipartite entangled pairs.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The N stations to which the source sends the entangled
        states, usually the neighboring repeater stations.
    label : str or None
        Optionally, provide a custom label.

    Attributes
    ----------
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The N stations to which the source sends the entangled
        states, usually the neighboring repeater stations.
    type : str
        "Source"

    """
     
    def __init__(self, world, position, target_stations, label=None):
        super().__init__(world, position, target_stations, label=None)
    

    def generate_multi_qubit(self, initial_state):
        """Generate an entangled states.

        The states will be generated in the `initial_state` at the
        `self.target_stations` of the source.
        Usually called from a SourceEvent.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial density matrix of the N-qubit

        Returns
        -------
        multi_qubit
            The newly generated states.

        """
        
        #check the size of the initial state density matrix
        size = initial_state.shape[0]
        N = np.log2(size))
        N = int(N)
        #potential sanity check: N should always be an integer


        # if 2 just use generated_pair and return 
        if N == 2:
            return super().generate_pair(self, initial_state)
        #else do for loop
        station = None
        qubits = []

        for i in range(N):
            station = self.target_stations[i]
            qubit.append(station.create_qubit())
        
        return mult_qubit(
            world=self.world, qubits=qubits, initial_state=initial_state
        )



class SchedulingSource(SourceMult):
    """A Source that schedules its next event according to a distribution.

    Parameters
    ----------
    world : World
        This WorldObject is an object in this world.
    position : scalar
        Position in meters in the 1D line for this linear repeater.
    target_stations : list of Stations
        The stations to which the source sends the entangled
        states, usually the neighboring repeater stations.
    time_distribution : callable
        Used for scheduling. Should return the amount of time until the next
        SourceEvent should take place (possibly probabilistic).
    state_generation : callable
        Should return (possibly probabilistically) the density matrix of the
        states generated by the source. Takes the source as input.
    label : str or None
        Optionally, provide a custom label.

    """

    def __init__(
        self,
        world,
        position,
        target_stations,
        time_distribution,
        state_generation,
        label=None,
    ):
        self.time_distribution = time_distribution
        self.state_generation = state_generation
        super(SchedulingSource, self).__init__(world, position, target_stations, label)

    def schedule_event(self):
        """Schedule a SourceEvent according to the specified rules.

        Will generate a pair after a time detemined by `time_distribution`
        and in a state specified by `state_generation`.

        Returns
        -------
        SourceEvent
            The event that was scheduled by this.

        """
        time_delay = self.time_distribution(source=self)
        scheduled_time = self.event_queue.current_time + time_delay
        initial_state = self.state_generation(
            source=self
        )  # should accurately describe state at the scheduled time
        source_event = events.SourceEvent(
            time=scheduled_time,
            source=self,
            initial_state=initial_state,
        )
        self.event_queue.add_event(source_event)
        return source_event

