from requsim.quantum_objects import Station
import numpy as np
from multiprocessing.connection import Connection
from multiprocessing import Queue
from datetime import datetime
from QC_aux_functions import angles


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
        self.rng = np.random.default_rng(
            seed=int(datetime.now().timestamp()) + actor_numb
        )  # some unique random number seed for each actor
        self.verifier = None

    def vote_init():
        self.rate = 0
        self.test_rate = 0

    def verify_or_vote(self, n, M, N, orderings, S):
        # receive a qubit from GHZ
        if n == self.order:
            # actor is agent
            r = self.rng.choice(2, 1, p=[1 - (2 ** (-1 * M)), 2 ** (-1 * M)])
            r_total, collision_r = self.logical_or(N=N, x_i=r, orderings=orderings, S=S)
            assert r == r_total
        else:
            r, collision_r = self.logical_or(N=N, x_i=0, orderings=orderings, S=S)
        assert collision_r is False
        if r == 0:
            # verification round
            return "verify"
        if r == 1:
            return "vote"

    def choose_verifier(self, n, N, orderings, S):
        K = np.int64(np.ceil(np.log2(N)))
        if n == self.actor_numb:
            # current agent
            ver = self.rng.choice(N, 1)[0] + 1
            ver = np.binary_repr(ver, K)
        else:
            ver = "0" * K
        ver_out = []
        for i in range(K):
            v_i, collision_v = self.logical_or(
                N=N, x_i=int(ver[i]), orderings=orderings, S=S
            )
            assert collision_v is False
            ver_out += [v_i]
        self.verifier = int("".join(map(str, ver_out)), 2)

    def verify(self, N, parent_conn):
        assert self.verifier is not None
        if self.verifier == self.actor_numb:
            self.test_rate += 1
            # create random angles
            m, angles_list = angles(N, self.rng)
            for i, conn in enumerate(self.connections):
                if conn is not None:
                    conn.send(angles_list[i])
            angle = angles_list[self.actor_numb]
        else:
            angle = self.connections[self.verifier].recv()
            # measure with the angle
        parent_conn.send(angle)
        meas_res = [parent_conn.recv()]
        # publicly announce, but for simulation it is sent to agent
        if self.verifier == self.actor_numb:
            for conn in self.connections:
                if conn is not None:
                    meas_res += conn.recv()
            if np.sum(meas_res) % 2 == m % 2:
                self.rate += 1
        else:
            self.connections[self.verifier].put(meas_result[0])
        self.verifier = None

    def get_unique_id(self, S, N, orderings):
        # implementation of the unique id classical subroutine from the POV of an actor
        omeg = 0
        for R in range(1, N):
            cont = True
            while cont:
                y_i = 0
                while y_i == 0:
                    if omeg != 0:
                        x_i = 0
                    else:
                        prob = 1 - (1 / (N + 1 - R))
                        x_i = self.rng.choice(2, 1, p=[prob, 1 - prob])[0]
                    y_i, collision = self.logic_or(
                        N=N, x_i=x_i, orderings=orderings, S=S
                    )
                assert y_i == 1
                c_i = 0
                if x_i == 1 and collision:
                    c_i = 1
                c_log, second_collision = self.logic_or(
                    N=N, x_i=c_i, orderings=orderings, S=S
                )
                if c_log == 0:
                    if x_i == 1:
                        omeg = R
                    cont = False
                # print(self.actor_numb, "R", R, "x", x_i, "c", c_i, "tru_coll", collision,"c_log", c_log ,"omeg", omeg)
        if omeg == 0:
            self.order = N
        else:
            self.order = omeg
        self.output_queue.put((self.actor_numb, omeg, self.order))

    def logic_or(self, N, x_i, orderings, S):
        # implementation of the logical or classical subroutine from the POV of an actor
        y_i = 0
        collision = False
        for ordering in orderings:
            for l in range(S):
                if x_i == 0:
                    p_i = 0
                else:
                    p_i = self.rng.choice(2, 1)[0]

                r_i = self.rng.choice(2, N - 1)
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
                z_j = np.sum(r_i_recv) + r_i[self.actor_numb] % 2
                # here we have to follow the fixed ordering
                # actor knows through ordering, who#s message he has to wait for
                # and who will broadcast the result at the end
                #
                # we add an extra z_final to check if there was a collision (see parity algorithm of logical or)
                if ordering[0] == self.actor_numb:
                    # here we need no check for none, because the list is created in such a way that self.connection[self.actor_numb] is None
                    self.connections[ordering[1]].send(z_j)
                    z_final = self.connections[ordering[-1]].recv()
                    y_i += z_final
                elif ordering[-1] == self.actor_numb:  # last in ordering
                    z_in = self.connections[ordering[-2]].recv()
                    z_final = (z_in + z_j) % 2
                    y_i += z_final
                    for conn in self.connections:
                        if conn is not None:
                            conn.send(z_final)
                else:  # not first or last, so find position
                    pos = np.argwhere(ordering == self.actor_numb)[0][0]
                    # wait to receive prior z
                    z_in = self.connections[ordering[pos - 1]].recv()
                    z_out = (z_in + z_j) % 2
                    self.connections[ordering[pos + 1]].send(z_out)
                    z_final = self.connections[ordering[-1]].recv()
                    y_i += z_final
                # check for collision
                if x_i == 1 and p_i == 0 and z_final == 1:
                    collision = True
                if p_i == 1 and z_final == 0:
                    collision = True
        # just a check
        self.output_queue.put((self.actor_numb, y_i, collision))
        return y_i > 0, collision
