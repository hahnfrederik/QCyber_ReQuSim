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

    def logic_or(self, N, x_i, ordering, S=1):
        # implementation of the logical or classical subroutine from the POV of an actor
        y_i = []
        for l in range(S):
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
                if conn is not None:
                    conn.send(r_i[j])
            r_i_recv = []
            for conn in self.connections:
                if conn is not None:
                    r_i_recv += [conn.recv()]
            z_j = np.sum(r_i_recv) % 2
            # here we have to follow the fixed ordering
            # actor knows through ordering, who#s message he has to wait for
            # and who will broadcast the result at the end
            if ordering[0] == self.actor_numb:
                # here we need no check for none, because the list is created in such a way that self.connection[self.actor_numb] is None
                self.connections[ordering[1]].send(z_j)
                y_i += [self.connections[ordering[-1]].recv()][0]
            elif ordering[-1] == self.actor_numb:  # last in ordering
                z_in = self.connections[ordering[-2]].recv()
                y_i += [(z_in + z_j) % 2]
                for conn in self.connections:
                    if conn is not None:
                        conn.send(y_i)
            else:  # not first or last, so find position
                pos = np.argwhere(ordering == self.actor_numb)[0][0]
                # wait to receive prior z
                z_in = self.connections[ordering[pos - 1]].recv()
                z_out = (z_in + z_j) % 2
                self.connections[ordering[pos + 1]].send(z_out)
                y_i += [self.connections[ordering[-1]].recv()][0]
        y_i = np.int64(np.sum(y_i) > 0)
        self.output_queue.put(y_i)
