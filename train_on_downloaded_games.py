from go import *
from neural_network import *
import h5py
import numpy as np
import threading
import random
import time


def get_input_ground_truth_pairs(game_history_file, game_number, move_number):
    """
    :param game_history_file:
    :param game_number:
    :param move_number: move number 1 is the first move of the game, played by black.
    :return:
    """

    # Then, get the keys to access the game.
    game_key = "game_" + str(game_number)

    # Next, get the outcome of the game.
    y_true_value = game_history_file[game_key]["outcome"][()]

    # Get the player to make the move.
    player_to_make_move = "black" if move_number % 2 == 0 else "white"

    # Get the board state, for the last 8 moves.
    last_eight_moves = []
    # First, get the last 8 boards.
    if move_number >= 8:
        last_eight_moves = game_history_file[game_key]["move_history"][move_number - 8:move_number, 0:169]
    else:
        for _ in range(abs(move_number - 8)):
            last_eight_moves.append([0] * 169)
        last_eight_moves.extend(game_history_file[game_key]["move_history"][0:move_number, 0:169])

    # Get the state for the last 8 moves
    board_state = get_full_state_from_byte_board_history(last_eight_moves, player_to_make_move)

    y_true_policy = [0] * 170
    mcts_selected_move = int(game_history_file[game_key]["move_history"][move_number - 1, -1])
    y_true_policy[mcts_selected_move] = 1

    return board_state, y_true_value, y_true_policy


def optimization(player_nn, mini_batch_num):
    """
    This function trains the neural network, using the move histories and game outcomes in the game history file.
    When training on 64 GPUs, each GPU used a minibatch size of 32, for a total minibatch size of 2048.
    Here, we will just use a minibatch size of 32.
    :param player_nn: a PolicyValueNetwork to train.
    :param mini_batch_num: an integer that shows which minibatch this is.
    """

    print("Optimizing on minibatch " + str(mini_batch_num))

    # Next, randomly select moves to train on.
    training_data = {"inputs": [], "y_true_values": [], "y_true_policies": []}

    games_added = 0
    while games_added < 2048:
        try:
            # Randomly select a game.
            game_range_low = 0
            game_range_high = 124393
            random_game = random.randint(game_range_low, game_range_high)

            # From that game, randomly select a move.
            game_key = "game_" + str(random_game)
            game_group = int((random_game + 1) / 1000)
            game_history_file = h5py.File("downloaded_game_data_" + str(game_group) + ".h5", "r")
            num_moves = game_history_file[game_key]["move_history"].shape[0]
            if num_moves < 50:
                continue
            move_range_low = 0
            #move_range_high = game_history_file[game_key]["move_history"].shape[0] - 1 # Train on all moves.
            move_range_high = 15 # Train on opening moves only.

            random_move = random.randint(move_range_low, move_range_high)

            # Add to the training data.
            input_state, output_value, output_policy = get_input_ground_truth_pairs(game_history_file,
                                                                                    random_game,
                                                                                    random_move)

            training_data["inputs"].append(input_state)
            training_data["y_true_values"].append(output_value)
            training_data["y_true_policies"].append(output_policy)
            games_added += 1

        except:
            print("failing to access game " + str(random_game))
            continue

    reshaped_inputs = np.reshape(np.array(training_data["inputs"]), (2048, 13, 13, 17))
    reshaped_gt_values = np.reshape(np.array(training_data["y_true_values"]), (2048, 1))
    reshaped_gt_policies = np.reshape(np.array(training_data["y_true_policies"]), (2048, 170))

    player_nn.train_supervised(reshaped_inputs,
                               reshaped_gt_values,
                               reshaped_gt_policies,
                               "young_saigon_supervised.h5")
    return 1


def optimization_loop_func():
    # Create the player.
    player_nn = PolicyValueNetwork(0.01, "young_saigon_supervised.h5")
    player_nn.save_model_to_file("young_saigon_supervised.h5")
    mini_batch_num = 0
    while True:
        mini_batch_num += optimization(player_nn, mini_batch_num)


optimization_loop_func()
