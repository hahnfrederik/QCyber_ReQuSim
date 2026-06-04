import numpy as np
from classical_Net.actor import Actor
from classical_Net.network import Network
from multiprocessing import Process, Queue, JoinableQueue

N = 3

X = [0, 0, 1]

network = Network(N)
actors = []
actor_process = []
output_queue = JoinableQueue(N)
for i in range(N):
    actors += [
        Actor(
            station=None,
            actor_numb=i,
            output_queue=output_queue,
            connections=network._get_conns(i),
        )
    ]
for i in range(N):
    ordering = np.random.permutation(N)
    for j, actor in enumerate(actors):
        curr_a_proc = Process(target=actor.logic_or, args=(N, X[j], ordering))
        curr_a_proc.daemon = True
        curr_a_proc.start()
        actor_process += [curr_a_proc]

    for ap in actor_process:
        ap.join()

    while not (output_queue.empty()):
        print(i, output_queue.get())
