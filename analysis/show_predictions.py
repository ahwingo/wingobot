"""
This utility shows the move selection probability distribution as a heat map on the command line.

WARNING: This file is technically LEGACY. It is out of date and will not currently work.
         However, it is a very useful tool for evaluating the bot's decision process.
         TODO: Get this utility up to date.
"""
# Python Standard
import argparse
import math
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import queue
import random
import sys
import time

# Third Party
import h5py
import numpy as np

# Local
sys.path.append("..")
from source.gooop import *
from source.nn_ll_tf import *
from source.training_library import *


# Load the bash heatmap colors. We will use this to display move selection probabilities.
with open("bash_heatmap.txt", "r") as f:
    heatmap_colors = [c.strip() for c in f.readlines()]


def print_heatmap(move_probs):
    legend = ""
    for color_code in heatmap_colors:
        legend += "\033[48;5;" + color_code + "m  \033[0m"
    print(f"Legend:\n{legend}\n")
    heatmap_bin_size = (move_probs.max() - move_probs.min()) / (len(heatmap_colors) - 2)
    for row in range(13):
        row_str = ""
        for col in range(13):
            move_idx = 13*row + col
            color_code = heatmap_colors[math.ceil(move_probs[0][move_idx] / heatmap_bin_size)]
            row_str += "\033[48;5;" + color_code + "m  \033[0m "
        print(row_str)


def print_board_local(black_board, white_board, bot_move_idx, act_move_idx):
    black = "16"
    white = "15"
    board = "222"
    bot_move = "9"
    act_move = "21"
    match_move = "10"
    for row in range(13):
        row_str = ""
        for col in range(13):
            if row*13 + col == act_move_idx and act_move_idx == bot_move_idx:
                val = match_move
            elif row * 13 + col == act_move_idx:
                val = act_move
            elif row*13 + col == bot_move_idx:
                val = bot_move
            elif black_board[row * 13 + col] == 1:
                val = black
            elif white_board[row * 13 + col] == 1:
                val = white
            else:
                val = board
            row_str += "\033[48;5;" + val + "m  \033[0m "
        print(row_str)


def prediction_loop(player_nn, trainer):
    """
    Loads games from the trainer, gets a random board state, and evaluates the network at that state.
    Prints the network output and waits for the user to press enter before continuing.
    """
    while True:
        inputs, gt_policies, gt_values = trainer.get_random_training_batch(trainer.registered_h5_files, 1, 8, 13)

        board_state_old = []
        for i in range(19):
            board_state_old.append(np.reshape(inputs, (169, 19))[...,i])
        player_to_move = "White" if board_state_old[16][0] == 0 else "Black"

        pred_moves, pred_value = player_nn.predict_given_state(inputs)
        pred_moves = np.asarray([np.random.dirichlet([0.03]*179)])
        #pred_moves = np.asarray([np.random.dirichlet(pred_moves[0])])
        bot_move_idx = np.argmax(pred_moves)
        act_move_idx = np.argmax(gt_policies[0])
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print("Player to Move: ", player_to_move)
        print("\nMove Probability Heat Map")
        print_heatmap(pred_moves)
        print("Sum Dir: ", pred_moves.sum())
        print("\nWin Likelihood: ", pred_value, " vs. Actual Result: ", gt_values)
        print("\nBoard State:")
        black_board = board_state_old[14] if player_to_move == "Black" else board_state_old[15]
        white_board = board_state_old[14] if player_to_move == "White" else board_state_old[15]
        print_board_local(black_board, white_board, bot_move_idx, act_move_idx)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        input()


def main():
    # Parse args.
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--botfile", default="../supervised_training/wingobot_sl_jun_7_2022_hlr/10_0.477.h5")
    parser.add_argument("-d", "--data_dir", default="../supervised_training/original_online_games/h5")
    args = parser.parse_args()

    # Get the library of games.
    trainer = TrainingLibrary()
    training_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith("h5")]
    for h5_file in training_files:
        trainer.register_h5_file(h5_file)

    # Create the Go bot.
    player_nn = PolicyValueNetwork(0.0001, starting_network_file=args.botfile)

    # Run the interactive (waits for user input before loading next example) prediction loop.
    prediction_loop(player_nn, trainer)


# Run the main function.
main()
