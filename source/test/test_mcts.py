from mcts_multiprocess import MonteCarloSearchTree as mcst
from threading import Thread
from queue import Queue
from gooop import Goban
import numpy as np

state_queue = Queue()
return_queue = Queue()


def move_selection_thread():
    while True:
        state = np.asarray([state_queue.get()["data"]["state"]])
        policy = np.random.random(170)
        value = np.random.random(1) - 0.5
        return_queue.put({"policy": policy, "value": value})

mst = Thread(target=move_selection_thread)
mst.start()

for move_num in range(168):
    search_tree = mcst(1, {"input": state_queue, "output": return_queue}, Goban(13))
    best_action = search_tree.search(32)
    if move_num % 2 == 0:
        print("black move at {}".format(best_action))
    else:
        print("white move at {}".format(best_action))



