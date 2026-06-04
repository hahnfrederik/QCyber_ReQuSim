from multiprocessing import Queue, Pipe


class Network:
    """classical netowrk class. This class will regulate the different channels (or in this case queues) and which quese belongs to what connection

    Parameters
    ----------
    N : int
        the size of the nwetowkr

    """

    def __init__(self, N):
        # initializing the pipes for communication
        self.N = N
        self.connections = [[] for i in range(N)]

        for i in range(N):
            for j in range(i + 1, N):
                conn1, conn2 = Pipe()
                self.connections[i] += [conn1]
                self.connections[j] += [conn2]

    def _get_conns(self, i):
        # get connections of the actor i
        return self.connections[i]

    def _close(self):
        for conns in self.connections:
            for conn in conns:
                conn.close()
        return
