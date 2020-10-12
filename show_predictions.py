import h5py
import math
import time
import queue
import random
from go import *
import numpy as np
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]=""
from nn_ll_tf import *


# Load the bash heatmap colors. We will use this to display move selection probabilities.
with open("bash_heatmap.txt", "r") as f:
    heatmap_colors = [c.strip() for c in f.readlines()]


def print_heatmap(move_probs):
    max_idx = np.argmax(move_probs)
    heatmap_bin_size = (move_probs.max() - move_probs.min()) / (len(heatmap_colors) - 2)
    for row in range(13):
        row_str = ""
        for col in range(13):
            move_idx = 13*row + col
            if move_idx == max_idx:
                color_code = heatmap_colors[-1]
            else:
                color_code = heatmap_colors[math.ceil(move_probs[0][move_idx] / heatmap_bin_size)]
            row_str += "\033[48;5;" + color_code + "m  \033[0m  "
        print(row_str)



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
    player_to_make_move = "black" if move_number % 2 == 1 else "white"
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

    #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    #print(" showing board states loaded from file...." )
    #print(int(game_history_file[game_key]["move_history"][move_number - 4, -1])//13," ", int(game_history_file[game_key]["move_history"][move_number - 4, -1])%13)
    #print_board(board_state[8], board_state[9])
    #print(int(game_history_file[game_key]["move_history"][move_number - 3, -1])//13," ",int(game_history_file[game_key]["move_history"][move_number - 3, -1])%13)
    #print_board(board_state[10], board_state[11])
    #print(int(game_history_file[game_key]["move_history"][move_number - 2, -1])//13," ",int(game_history_file[game_key]["move_history"][move_number - 2, -1])%13)
    #print_board(board_state[12], board_state[13])
    #print(int(game_history_file[game_key]["move_history"][move_number - 1, -1])//13," ",int(game_history_file[game_key]["move_history"][move_number - 1, -1])%13)
    #print_board(board_state[14], board_state[15])
    #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    # Check if any of the moves has a negative value...
    moves = [int(game_history_file[game_key]["move_history"][move_number - x, -1]) for x in [1,2,3,4]]
    for mov in moves:
        if mov < 0:
            print("A move was negative....")
            exit()


    # Add the liberty counts to the board state.
    friendly_value = 1 if player_to_make_move is "black" else 2
    current_board = game_history_file[game_key]["move_history"][move_number-1, 0:169]
    friendly, enemy = get_liberty_counts_from_board(current_board, friendly_value)

    """
    print("\nFriendly Lib Count Board: \n")
    for r in range(13):
        r_str = ""
        for c in range(13):
            r_str += str(friendly[r*13 + c]) + " "
        print(r_str)

    print("\nEnemy Lib Count Board: \n")
    for r in range(13):
        r_str = ""
        for c in range(13):
            r_str += str(enemy[r * 13 + c]) + " "
        print(r_str)
    """

    board_state.append(friendly)
    board_state.append(enemy)

    y_true_policy = [0] * 170
    mcts_selected_move = int(game_history_file[game_key]["move_history"][move_number - 1, -1])


    if mcts_selected_move < 0:
        print(mcts_selected_move)
        print(game_history_file[game_key]["move_history"][move_number - 1, -1])
        print(type(game_history_file[game_key]["move_history"]))
        print("jhj;kljl;kjl;kjlkj;lk \n\n\n\n asfdasdfasdfasdfasdfasdf \n\n\n\n\n")
        exit()

    y_true_policy[mcts_selected_move] = 1

    return board_state, y_true_value, y_true_policy


def get_random_game_move_data():
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
    while games_added < 1:
        try:
            # Randomly select a game.
            game_range_low = 1000
            game_range_high = 117000
            random_game = random.randint(game_range_low, game_range_high)

            # From that game, randomly select a move.
            game_key = "game_" + str(random_game)
            game_group = int((random_game + 1) / 1000)
            game_history_file = h5py.File("downloaded_game_data/downloaded_game_data_" + str(game_group) + ".h5", "r")
            num_moves = game_history_file[game_key]["move_history"].shape[0]
            if num_moves < 50:
                continue
            move_range_low = 1
            move_range_high = game_history_file[game_key]["move_history"].shape[0] - 1 # Train on all moves.
            #move_range_high = 15 # Train on opening moves only.

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

        except:
            print("failing to access game " + str(random_game))
            continue


    #print("np.array.shape = ", np.array(training_data["inputs"]).shape)


    #reshaped_inputs = np.array([np.reshape(np.array(x), (13,13)) for x in training_data["inputs"][0]])
    width_height_depth = [[training_data["inputs"][0][j][i] for j in range(19)] for i in range(169)]
    reshaped_inputs = np.reshape(np.array(width_height_depth), (1, 19, 13, 13))
    #print(reshaped_inputs.shape)
    reshaped_gt_values = np.reshape(np.array(training_data["y_true_values"]), (1, 1))
    reshaped_gt_policies = np.reshape(np.array(training_data["y_true_policies"]), (1, 170))

    #print("---------------------------------")
    #print(" training data inputs 5 = ")
    #print(training_data["inputs"][0][10])

    #print("---------------------------------")
    #print(" reshaped training data inputs 5 = ")
    #print(reshaped_inputs[0,...,10])
    #print("---------------------------------")

    return {"inputs": reshaped_inputs, "gt_values": reshaped_gt_values, "gt_policies": reshaped_gt_policies}



def prediction_loop(player_nn):
    while True:
        training_data = get_random_game_move_data()
        if training_data:
            inputs = training_data["inputs"]
            gt_values = training_data["gt_values"]
            gt_policies = training_data["gt_policies"]
            board_state_old = []
            for i in range(19):
                board_state_old.append(np.reshape(inputs, (169, 19))[...,i])
            player_to_move = "White" if board_state_old[16][0] == 0 else "Black"
            legal_moves = get_all_legal_moves_from_board_state(board_state_old[:17])
            legal_moves.append(1)
            legal_moves = np.asarray([legal_moves])
            prior_probs, pred_value = player_nn.predict_given_state(inputs)
            best_legal_moves = legal_moves*prior_probs
            action_idx = np.argmax(best_legal_moves)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("Player to Move: ", player_to_move)
            print("\nMove Probability Heat Map")
            print_heatmap(best_legal_moves)
            print("\nWin Likelihood: ", pred_value, " vs. Actual Result: ", gt_values)
            print("\nCurrent board state:")
            black_board = board_state_old[14] if player_to_move == "Black" else board_state_old[15]
            white_board = board_state_old[14] if player_to_move == "White" else board_state_old[15]
            print_board(black_board, white_board)
            print("\nBoard after move at row ", action_idx//13, " column ", action_idx % 13, ":")
            board_state = update_board_state_for_move(action_idx, board_state_old[:17])
            black_board = board_state[15] if player_to_move == "Black" else board_state[14]
            white_board = board_state[15] if player_to_move == "White" else board_state[14]
            print_board(black_board, white_board)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            input()


def main():
    # Create the player.
    #player_nn = PolicyValueNetwork(0.0001, starting_network_file="young_ryzen_ckpt_30500.h5")
    player_nn = PolicyValueNetwork(0.0001, starting_network_file="young_thread_ripper_ckpt_11500.h5")

    # Run the prediction loop.
    prediction_loop(player_nn)


# Run the main function.
main()
