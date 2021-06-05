import os
import h5py
import random
import signal
import argparse
import numpy as np
from time import time
from gooop import Goban
import multiprocessing as mp
from mcts_multiprocess import MonteCarloSearchTree
from player_controller import PlayerController, GoBotTrainer

# Initialize the Goban.
go_board = Goban(13)

def attempt_move(row, col):
    """
    Try making a move and get the updated board state. Reset all stones on the display.
    """
    go_board.make_move(row, col)
    go_board.print_board()

# Initialize the player controller stuff.
queue_manager = mp.Manager()
bot_1_input_queue = queue_manager.Queue()
bot_1_output_queue = queue_manager.Queue()
bot_1_output_queues_map = {0: bot_1_output_queue}
bot_1_processing_queue = PlayerController("shodan_fossa/shodan_focal_fossa_109.h5", 1, bot_1_input_queue)
bot_1_processing_queue.start()
bot_1_processing_queue.set_output_queue_map(bot_1_output_queues_map)
bot_1_queues = {"input": bot_1_input_queue, "output": bot_1_output_queue}
black_search_tree = MonteCarloSearchTree(0, bot_1_queues, go_board)
num_simulations = 32
for move_num in range(128 // 2):
    # Black makes a move.
    best_black_move = black_search_tree.search(num_simulations)
    black_move_row = best_black_move // 13
    black_move_col = best_black_move % 13
    print("Black move: ", black_move_row, " ", black_move_col)
    attempt_move(black_move_row, black_move_col)

    # White makes a move.
    print("Row: ")
    row = int(input())
    print("Col: ")
    col = int(input())
    best_white_move = row*13 + col
    attempt_move(row, col)
    if best_white_move in black_search_tree.root.children:
        black_search_tree.update_root(black_search_tree.root.children[best_white_move])
    else:
        black_search_tree = MonteCarloSearchTree(0, bot_1_queues, go_board.copy())
