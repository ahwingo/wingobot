import numpy as np
import h5py
from go import *


"""
Game outcomes stored as 1 for black wins, -1 for white wins

Boards saved as 0 empty 1 black 2 white

"""

# Get the index given the alphabet values of the move.
move_code_map = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
                 "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12}


def save_game_data(game_num, game_outcome, move_history, h5_file):
    """
    This function saves a games history of moves and its outcome into an HDF5 file.
    :param game_num: the integer used to identify this game.
    :param game_outcome: the outcome of the game (1 if black wins, -1 if white wins).
    :param move_history: a list of arrays that holds the state of the board, moving player, and move index.
    :param h5_file: the HDF5 file that data is stored in.
    """
    # Next, create a group for this game.
    game_group = h5_file.create_group("game_" + str(game_num))
    # Create a data set for the games outcome.
    game_group.create_dataset("move_history", move_history.shape, dtype=move_history.dtype, data=move_history)
    # Create a data set for the winner of the game.
    game_group.create_dataset("outcome", game_outcome.shape, dtype=game_outcome.dtype, data=game_outcome)


def get_index_for_move_code(move_code):
    global move_code_map
    move_row = move_code_map[move_code[0]]
    move_col = move_code_map[move_code[1]]
    return 13 * move_row + move_col


def load_downloaded_games():

    # Open the H5 file that will be used.
    game_history_file = h5py.File("downloaded_game_data.h5", 'w')

    # Open the file that contains all the paths to downloaded sgf files.
    with open("Go_Games_13x13/files.txt", "r") as f:
        file_paths = f.readlines()
    print(len(file_paths))

    # Open each SGF file and read the moves. Store moves and outcomes into an HDF5 file.
    game_number = 0
    for sgf_file in file_paths:

        try:
            sgf_file = sgf_file.strip()

            # Get the content from this file.
            with open(sgf_file, "r") as sgf_f:
                content = sgf_f.readlines()

            # Get the result of the game.
            winner = content[2][3+content[2].find("RE[")]
            game_outcome = 1 if winner is "B" else -1
            game_outcome = np.array(game_outcome, dtype='b')

            # Get all the moves. Default move value is 169 for a pass.
            moves = []
            move_value = 169
            moves_line_by_line = content[3:]
            moves_concatenated = ""
            for move_line in moves_line_by_line:
                moves_concatenated += move_line
            moves_split = moves_concatenated.split(";")[1:]
            for move_string in moves_split:
                if move_string[4] is "]":
                    move_code = move_string[2:4]
                    move_value = get_index_for_move_code(move_code)
                moves.append(move_value)

            # Initialize the state of the empty board from blacks perspective.
            board_state = []
            for _ in range(16):
                board_state.append([0]*169)
            board_state.append([1]*169)

            # Maintain a history of moves and resulting board states.
            move_history = []

            # Play the game out and store histories.
            for idx in range(0, len(moves), 2):

                # Black moves.
                action_idx = moves[idx]

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

                # Make sure there are moves remaining.
                if idx + 1 >= len(moves):
                    break

                # White moves
                action_idx = moves[idx + 1]

                # Store the state and action.
                current_board_and_chosen_action = get_single_storable_board_from_state(board_state[15], board_state[14])
                current_board_and_chosen_action.append(2)
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

            # Convert the move history array to a np array.
            move_history = np.array(move_history, dtype='b')

            # Store the outcome of the game.
            save_game_data(game_number, game_outcome, move_history, game_history_file)

            # Increment the number of games.
            game_number += 1
            if game_number % 100 is 0:
                print("Done with game number " + str(game_number))

        except:
            continue


# Run the main code.
load_downloaded_games()
