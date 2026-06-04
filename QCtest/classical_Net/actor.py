from requsim.quantum_objects import Station
import numpy as np
from multiprocessing.connection import Connection
from multiprocessing import Queue


class Actor:
    """Actor class, for each Actor in the Newtork.
    This class should include all the actions an actor can do in the network and also do the local calculation each actor is required to do.

    Parameters
    ----------
    station: Station
        the requsim station that represents this actor
    actor_numb:
        the actor number with which the actor is identified in the network
    network: Network
        the network

    Attributes
    ----------
    unique_id : int
        the unique id received by the unique id protocol
    turn : boolean
        true if currently the voting agent

    """

    def __init__(self, station: Station, actor_numb, output_queue, connections):
        self.station = station
        self.actor_numb = actor_numb
        self.unique_id = None
        self.turn = False
        self.output_queue = output_queue
        self.connections = connections

    def get_unique_id(self, queues, R, N):
        # implementation of the unique id classical subroutine from the POV of an actor
        omeg = 0
        if omeg == 1:
            x_i = 0
        else:
            prob = 1 - (1 / (N - R))
            x_i = np.random.choice(2, 1, p=[prob, 1 - prob])[0]

    def logic_or(self, N, x_i):
        # implementation of the logical or classical subroutine from the POV of an actor

        if x_i == 0:
            p_i = 0
        else:
            p_i = np.random.choice(2, 1)[0]

        r_i = np.random.choice(2, N - 1)
        if np.sum(r_i) % 2 == p_i:
            r_i = np.append(r_i, 0)
        else:
            r_i = np.append(r_i, 1)

        for j, conn in enumerate(self.connections):
            conn.send(r_i)
        r_i_recv = []
        for conn in self.connections:
            r_i_recv += [conn.recv()]
        z_j = np.sum(r_i_recv) % 2
        for conn in self.connections:
            conn.send(z_j)
        z_s = []
        for conn in self.connections:
            z_s += [conn.recv()]

        self.output_queue.put((np.sum(z_s) + z_j) % 2)
