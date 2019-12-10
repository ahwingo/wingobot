"""
Objective:
Train neural networks to play Go until they are really, really good. Have 64 games being played all at once.

Parts:
    -   Self Play: Have the current network play against itself to generate moves.
    -   Optimization: Randomly sample from the last 500,000 moves, training the network in mini batches of 8.
    -   Evaluation: Every X games, compare the current network to the previous best (to ensure the network is learning).
"""

import mcts_lib_layers
from go import *
from neural_network_lib_layers import *
import h5py
import numpy as np
import threading
import random
import time

# GLOBALS

results_ready = False

batch_inputs_lock = threading.Lock()
batch_inputs = []

batch_outputs_lock = threading.Lock()
batch_outputs = []

num_outputs_recieved_lock = threading.Lock()
num_outputs_recieved = 0

"""
TODO: Determine what this function was going to be used for...
def predict_on_batch(neural_net, batch_size):
    global results_ready
    global batch_inputs
    global batch_outupts
    global num_outputs_recieved
    while True:
        results_ready = False
        if len(batch_inputs) >= batch_size:
            input_data = batch_inputs[:batch_size]
            batch_inputs = batch_inputs[batch_size:]
            input_data_as_ndarray = np.reshape(input_data, (batch_size, 13, 13, 19))
            results = neural_net.predict_given_state(input_data_as_ndarray, batch_size=batch_size)
            batch_outputs = results
            results_ready = True
        if results_ready:
            while num_outputs_recieved < batch_size:
                continue
            results_ready = False
            batch_outputs = []
            num_outputs_recieved = 0
"""


def save_game_data(game_num, game_outcome, move_history, h5_file):
    """
    This function saves a games history of moves and its outcome into an HDF5 file.
    :param game_num: the integer used to identify this game.
    :param game_outcome: the outcome of the game (1 if black wins, -1 if white wins).
    :param move_history: a list of arrays that holds the state of the board, moving player, and move index.
    :param output_filename: the HDF5 file that data is stored in.
    """
    # Next, create a group for this game.
    game_group = h5_file.create_group("game_" + str(game_num))
    # Create a data set for the games outcome.
    game_group.create_dataset("move_history", move_history.shape, dtype=move_history.dtype, data=move_history)
    # Create a data set for the winner of the game.
    game_group.create_dataset("outcome", game_outcome.shape, dtype=game_outcome.dtype, data=game_outcome)


def self_play(player, game_number, output_file):
    """
    This function hosts a single game of self play, using the player neural network. It stores moves in the output file.
    :param player: a PolicyValueNetwork that will play against itself, in a stable (non training) state.
    :param game_number: a unique number to identify which game is being played.
    :param output_filename: the name of an HDF5 file that the stores the moves, board states, and final outcome.
    """
    # Time each game.
    start_time = time.time()

    # Keep track of these to know when we can end the game.
    num_consecutive_passes = 0
    total_moves = 0
    move_history = []

    # Initialize the state of the empty board from blacks perspective.
    board_state = []
    for _ in range(16):
        board_state.append([0]*169)
    board_state.append([1]*169)

    # Keep track of the roots, so that we can save search tree history.
    black_root_node=None
    white_root_node=None

    # While the game is not over, select moves for each player using MCTS.
    while num_consecutive_passes < 2 and total_moves < 169:

        #print("Move number " + str(total_moves) + ".")

        temperature = 1.0 if total_moves < 60 else 0.2

        # Black moves.
        black_mcst = mcts_lib_layers.MonteCarloSearchTree(player, board_state, temperature, root=black_root_node)
        action_idx, black_root_node = black_mcst.search(50)
        total_moves += 1

        # Store the state and action.
        current_board_and_chosen_action = get_single_storable_board_from_state(board_state[14], board_state[15])
        current_board_and_chosen_action.append(1)
        current_board_and_chosen_action.append(action_idx)
        move_history.append(current_board_and_chosen_action)

        # If the selected move is a pass, increment the number of consecutive passes.
        if action_idx == 169:
            num_consecutive_passes += 1
            if num_consecutive_passes == 2:
                break
        else:  # Reset the number of passes.
            num_consecutive_passes = 0

        # Update the board states, now that a move has been made.
        board_state = update_board_state_for_move(action_idx, board_state)

        # Display the board.
        #print_board(board_state[7], board_state[6])

        # White moves.
        white_mcst = mcts_lib_layers.MonteCarloSearchTree(player, board_state, temperature, root=white_root_node)
        action_idx, white_root_node = white_mcst.search(50)
        total_moves += 1

        # Store the state and action.
        current_board_and_chosen_action = get_single_storable_board_from_state(board_state[15], board_state[14])
        current_board_and_chosen_action.append(1)
        current_board_and_chosen_action.append(action_idx)
        move_history.append(current_board_and_chosen_action)

        # If the selected move is a pass, increment the number of consecutive passes.
        if action_idx == 169:
            num_consecutive_passes += 1
            if num_consecutive_passes == 2:
                break
        else:  # Reset the number of passes.
            num_consecutive_passes = 0

        # Update the board states, now that a move has been made.
        board_state = update_board_state_for_move(action_idx, board_state)

        # Display the board.
        #print_board(board_state[14], board_state[15])

    # Display the board.
    print("Final board state.")
    print_board(board_state[14], board_state[15])

    # At this point, the game is over. Calculate the score and update the networks.
    if board_state[16][0] == 1:
        board = get_single_scorable_board_from_state(board_state[14], board_state[15])
    else:
        board = get_single_scorable_board_from_state(board_state[15], board_state[14])
    game_outcome = 1 if tromp_taylor_score(board) > 7.5 else -1

    # Convert the values to numpy arrays.
    move_history = np.array(move_history, dtype='B') # NOTE: Use unsigned byte (B) to avoid overflow error.
    game_outcome = np.array(game_outcome, dtype='b') # Okay to use signed byte (b) here.

    # Store the outcome of the game.
    save_game_data(game_number, game_outcome, move_history, output_file)

    # The end times.
    end_time = time.time()
    print("Game completed in " + str(end_time - start_time) + " seconds.")


def self_play_loop_func(game_history_file, player_nn, num_self_play_events):
    # First, get the total number of games that have been played so far.
    total_number_of_games_played = len(list(game_history_file.keys()))

    for game in range(num_self_play_events):
        print("Playing game " + str(total_number_of_games_played) + ".")
        self_play(player_nn, total_number_of_games_played, game_history_file)
        total_number_of_games_played += 1


def get_input_ground_truth_pairs(game_history_file, game_number, move_number):
    """

    :param game_history_filename:
    :param game_number:
    :param move_number: move number 1 is the first move of the game, played by black.
    :return:
    """

    # Get the player to make the move.
    player_to_make_move = "black" if move_number % 2 == 0 else "white"

    # Then, get the keys to access the game.
    game_key = "game_" + str(game_number)

    # Next, get the outcome of the game.
    y_true_value = game_history_file[game_key]["outcome"][()] # 1 if black wins, else -1 for white
    if player_to_make_move == "white":
        y_true_value *= -1

    # Get the board state, for the last 8 moves.
    last_eight_moves = []
    # First, get the last 8 boards.
    if move_number >= 8:
        last_eight_moves = game_history_file[game_key]["move_history"][move_number-8:move_number, 0:169]
    else:
        for _ in range(abs(move_number - 8)):
            last_eight_moves.append([0]*169)
        last_eight_moves.extend(game_history_file[game_key]["move_history"][0:move_number, 0:169])

    # Get the state for the last 8 moves
    board_state = get_full_state_from_byte_board_history(last_eight_moves, player_to_make_move)

    # Append the liberties.
    friendly_value = 1 
    friendly, enemy = get_liberty_counts_from_board(last_eight_moves[-1], friendly_value)
    board_state.append(friendy)
    board_state.append(enemy)

    y_true_policy = [0]*170
    mcts_selected_move = int(game_history_file[game_key]["move_history"][move_number-1, -1])
    y_true_policy[mcts_selected_move] = 1

    return board_state, y_true_value, y_true_policy


def optimization(game_history_file, player_nn, mini_batch_num):
    """
    This function trains the neural network, using the move histories and game outcomes in the game history file.
    When training on 64 GPUs, each GPU used a minibatch size of 32, for a total minibatch size of 2048.
    Here, we will just use a minibatch size of 32.
    :param game_history_filename: an HDF5 file that holds game histories.
    :param player_nn: a PolicyValueNetwork to train.
    :param minibatch_num: an integer that shows which minibatch this is.
    """

    # Get the total number of games that have been played.
    total_num_games = len(list(game_history_file.keys()))

    # If fewer than 5 games have been played, return.
    if total_num_games < 5:
        return 0

    print("Optimizing on minibatch " + str(mini_batch_num))

    # Next, randomly select moves to train on.
    training_data = {"inputs": [], "y_true_values": [], "y_true_policies": []}

    for _ in range(2048):
        # Randomly select a game.
        game_range_low = max(0, total_num_games - 500000)
        game_range_high = total_num_games - 1  # -1 because range is inclusive
        random_game = random.randint(game_range_low, game_range_high)

        # From that game, randomly select a move.
        game_key = "game_" + str(random_game)
        move_range_low = 1
        move_range_high = game_history_file[game_key]["move_history"].shape[0]
        random_move = random.randint(move_range_low, move_range_high)

        # Add to the training data.
        input_state, output_value, output_policy = get_input_ground_truth_pairs(game_history_filename,
                                                                                random_game,
                                                                                random_move)

        training_data["inputs"].append(input_state)
        training_data["y_true_values"].append(output_value)
        training_data["y_true_policies"].append(output_policy)

    reshaped_inputs = np.reshape(np.array(training_data["inputs"]), (2048, 13, 13, 19))
    reshaped_gt_values = np.reshape(np.array(training_data["y_true_values"]), (2048, 1))
    reshaped_gt_policies = np.reshape(np.array(training_data["y_true_policies"]), (2048, 170))

    player_nn.train(reshaped_inputs,
                    reshaped_gt_values,
                    reshaped_gt_policies)
    return 1


def optimization_loop_func(game_history_filename, player_nn):
    mini_batch_num = 0
    while True:
        mini_batch_num += optimization(game_history_filename, player_nn, mini_batch_num)
    pass


def main(game_history_filename, starting_network_file=None, best_network_file=None):
    """
    This main function manages the three stages of the training pipeline.
    :param game_history_filename: the HDF5 file that game histories are stored in.
    :param starting_network_file: the previous weights of the network.
    :param best_network_file: the previous weights of the network.
    """

    #Load the game history file. Pass this file to the functions.
    game_history_file = h5py.File(game_history_filename, 'a')

    # Load the starting neural network.
    player_nn = PolicyValueNetwork(0.01, starting_network_file)

    # Set up the pipeline loops.
    self_play_loop = threading.Thread(target=self_play_loop_func, args=(game_history_file, player_nn, 100000))
    optimization_loop = threading.Thread(target=optimization_loop_func, args=(game_history_file, player_nn))

    # Play games, optimize neural networks, and evaluate progress.
    self_play_loop.run()
    #optimization_loop_func(game_history_file, player_nn)
    game_history_file.close()

main("game_history_pretrained_acc26_dec_8.h5", starting_network_file="young_goon_sl_acc26_rl_init.h5")

