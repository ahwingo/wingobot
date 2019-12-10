from go import *
from neural_network_lib_layers import *
import h5py
import numpy as np
import threading
import random
import time
import queue

# Define a global Queue
training_batches = queue.Queue()

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
    # Set y_true_value to reflect the player making the move (stored in database as outcome relative to black player).
    if player_to_make_move is "white":
        y_true_value *= -1

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

    # Add the liberty counts to the board state.
    friendly_value = 1 if player_to_make_move is "black" else 2
    current_board = game_history_file[game_key]["move_history"][move_number-1, 0:169]
    friendly, enemy = get_liberty_counts_from_board(current_board, friendly_value)
    board_state.append(friendly)
    board_state.append(enemy)

    y_true_policy = [0] * 170
    mcts_selected_move = int(game_history_file[game_key]["move_history"][move_number - 1, -1])
    y_true_policy[mcts_selected_move] = 1

    return board_state, y_true_value, y_true_policy


def get_2048_training_batch(num_calls):
    """
    This function trains the neural network, using the move histories and game outcomes in the game history file.
    When training on 64 GPUs, each GPU used a minibatch size of 32, for a total minibatch size of 2048.
    Here, we will just use a minibatch size of 32.
    :param player_nn: a PolicyValueNetwork to train.
    :param mini_batch_num: an integer that shows which minibatch this is.
    """
    # Next, randomly select moves to train on.
    training_data = {"inputs": [], "y_true_values": [], "y_true_policies": []}

    games_added = 0
    while games_added < 2048:
        try:
            # Randomly select a game.
            game_range_low = 1000
            game_range_high = 117000
            random_game = random.randint(game_range_low, game_range_high)

            # From that game, randomly select a move.
            game_key = "game_" + str(random_game)
            game_group = int((random_game + 1) / 1000)
            game_history_file = h5py.File("downloaded_game_data_new/downloaded_game_data_" + str(game_group) + ".h5", "r")
            num_moves = game_history_file[game_key]["move_history"].shape[0]
            if num_moves < 50:
                continue
            move_range_low = 1
            # Train on the first X moves only.
            #first_x_moves = 16  # Consider only opening moves.
            first_x_moves = min(8 + num_calls // 50, 100)
            total_moves = game_history_file[game_key]["move_history"].shape[0] - 1
            move_range_high = min(total_moves, first_x_moves)
            random_move = random.randint(move_range_low, move_range_high)

            # Add to the training data.
            input_state, output_value, output_policy = get_input_ground_truth_pairs(game_history_file,
                                                                                    random_game,
                                                                                    random_move)
            # Close the game history file.
            game_history_file.close()

            training_data["inputs"].append(input_state)
            training_data["y_true_values"].append(output_value)
            training_data["y_true_policies"].append(output_policy)
            games_added += 1
            print("here")

        except:
            print("failing to access game " + str(random_game))
            continue

    #width_height_depth = [[[training_data["inputs"][k][j][i] for j in range(19)] for i in range(169)] for k in range(2048)]
    reshaped_inputs = np.reshape(np.array(training_data["inputs"]), (2048, 13, 13, 19))
    reshaped_gt_values = np.reshape(np.array(training_data["y_true_values"]), (2048, 1))
    reshaped_gt_policies = np.reshape(np.array(training_data["y_true_policies"]), (2048, 170))

    return {"inputs": reshaped_inputs, "gt_values": reshaped_gt_values, "gt_policies": reshaped_gt_policies}


def data_prep_thread(thread_num):
    num_calls = 0
    while True:
        num_calls += 1
        training_data = get_2048_training_batch(num_calls)
        training_batches.put(training_data)


def optimization_loop(player_nn):
    mini_batch_num = 0
    while True:
        training_data = training_batches.get()
        if training_data:
            print("Training on mini batch ", mini_batch_num)
            inputs = training_data["inputs"]
            gt_values = training_data["gt_values"]
            gt_policies = training_data["gt_policies"]
            player_nn.train_supervised(inputs, gt_values, gt_policies, "young_opener.h5")
            mini_batch_num += 1

            # Save a checkpoint every 500 mini batches.
            if mini_batch_num % 500 == 0:
                ckpt_file = "young_stair_climber_ckpt_" + str(mini_batch_num) + ".h5"
                player_nn.save_checkpoint(ckpt_file)


def main():
    # Create the player.
    player_nn = PolicyValueNetwork(0.0001, train_supervised=True, starting_network_file="young_opener.h5")
    player_nn.save_model_to_file("young_stair_climber.h5")

    # Create and run a handful of data prep threads.
    for x in range(1):
        dp_thread = threading.Thread(target=data_prep_thread, args=(x,))
        dp_thread.start()

    # Run the optimzation loop on this thread.
    optimization_loop(player_nn)

# Run the main function.
main()
