"""
    Voting round simulation

    Goal: Express anonymously the preference of the voting
        agent a.
    Inputs: The subround counter (p,k,n), that identifies the
        voting agent a; the k-th bit of the input s_a in [C]
        represented in binary units, s_q^k in {0,1}
    Outputs:
        The result of a subround, which is a N
        dimensional vector that will be part of the bulletin board
        B, more precisely the n-th row of the bulletin of the (p,k)
        subround.
"""

import numpy as np
from requsim.world import World
from requsim.quantum_objects import Station, SourceMult, MultiQubit
from requsim.events import MultiSourceEvent, MeasurementEvent
from requsim.libs.aux_functions import distance
import requsim.libs.matrix as mat

speedMeas = 1e-9  # speed of one quantum measurement
C = 2e8  # speed of light


class Voting:
    """
    Voting class, one class per vote.
    Attributes
        tally : np.darray, Tally of the Vote
        P : int, Privacy amplification round P (maximum and current) ((current could be read from the current state of the tally))
        C : int, Number of Candidates C
        N : int, Number of Voters N
        world : World, requsim world within which we are simulating
        stations: list[Station], the stations representing the
        source: SourceMult, the source of the ghz states for the protocol
    """

    def __init__(self, P, C, N, world, stations, source):
        self.tally = [[[] for j in range(np.ceil(np.log2(C)))] for i in range(P)]
        self.P = P
        self.N = N
        self.C = C
        self.world = world
        self.stations = stations
        self.source = source
        self.delay = np.max(
            [distance(source, station) / C for station in self.stations]
        )
        self.current_subround = (0, 0, 0)  # (p,k,n)

    def unique_index():
        # here, theoretically, we could do the unique index classical subroutine
        # for now we just return standard index list
        return np.arange(self.N)

    def send_ghz():
        # send the ghz state to all participants
        sending_ghz = MultiSourceEvent(
            time=self.world.event_queue.current_time + self.delay,
            source=self.source,
            initial_state=mat.ghz(self.N) @ mat.H(mat.ghz(self.N)),
        )
        world.event_queue.add_event(sending_ghz)

        result = world.event_queue.resolve_next_event()
        return result

    def increase_subrounds():
        p = self.current_subround[0]
        k = self.current_subround[1]
        n = self.currnet_subround[2]
        if n < self.N:
            self.current_subround = (p, k, n + 1)
        else:
            if k < np.ceil(np.log2(self.C)):
                self.current_subround = (p, k + 1, 0)
            else:
                if p < self.P:
                    self.current_subround = (p + 1, 0, 0)
                else:
                    # was last round
                    return -1
        return 0

    def measure_ghz(result):
        # each agent measures
        meas_result = []
        for station in self.stations:
            event_measure = MeasurementEvent(
                time=worldevent_queue.current_time + speedMeas,
                multiqubit=result["output_state"],
                station=station,
            )
            world.event_queue.add_event(event_measure)
            result = world.event_queue.resolve_next_event()
            meas_result += [result["measurment_outcome"]]
        return meas_result

    def vote_subround(station_a, meas_results):
        # station_a is the main station in this subround
        result = self.send_ghz()
        outcomes = self.measure_ghz(result)
        ntallyrow = []

        # assuming here that meas_result has same indexing as self.stations
        for i, station in enumerate(self.stations):
            if station == station_a:
                # station a checks if it is last round of privacy amplification
                if self.current_subround < self.P - 1:
                    # generat_random_bit
                    rand = np.random.randint(0, 2)
                else:
                    # the kth bit of the candidate agent a wants to vote for
                    # here we also do random for now
                    # in future maybe implement some way of input
                    rand = np.random.randint(0, 2)
            else:
                rand = np.random.randint(0, 2)
            ntallyrow += [rand + meas_result[i]]

        # every agent broadcasts

        tally[self.current_subround[0]][self.current_subround[1]] += [ntallyrow]
        self.increase_subrounds()

        return ntallyrow
