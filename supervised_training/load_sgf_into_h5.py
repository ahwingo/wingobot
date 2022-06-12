"""
This script loads the SGF files defined in a list (e.g. from original_online_games.txt)
into a set of H5 files stored in the given output directory.

  * The structure of the h5 file should be:

    - total_games: int
    - games:
          - game_0:
                - outcome: "W+" or "B+"
                - num_moves: integer
                - moves: nparray of shape (num_moves, 2) holding the [row, col] of each move. even==black odd==white
                - white_liberties: nparray of shape (num_moves, 13, 13)
                - black_liberties: nparray of shape (num_moves, 13, 13)
                - black_states: nparray of shape (num_moves, 13, 13)
                - white_states: nparray of shape (num_moves, 13, 13)
          .
          .
          .
          - game_128:

Game outcomes stored as 1 for black wins, -1 for white wins
Boards saved as 0 empty 1 black 2 white
"""
import argparse
from multiprocessing import Pool
import h5py
import sys
sys.path.append("..")
from source.gooop import *
import os

class SGF2H5Writer:
    """
    Instances of this class will write SGF data to an H5 file.
    """
    def __init__(self, sgf_files, output_dir, games_per_file, name="orig_online_games", history=8, board_size=13):
        # Determine how many H5 files to create.
        self.h5_file_groups = {}
        index = 0
        remaining = len(sgf_files)
        count = 0
        while remaining > 0:
            start_idx = index
            index += min(len(sgf_files) - start_idx, games_per_file)
            h5_group = sgf_files[start_idx:index]
            h5_path = os.path.join(output_dir, name + "_" + str(count) + ".h5")
            self.h5_file_groups[h5_path] = h5_group
            count += 1
            remaining -= index - start_idx

        # Store information relevant to the game storage.
        self.board_size = board_size
        self.history = history

    @staticmethod
    def load_sgf_into_data(data):
        sgf_file, board_size, history = data
        goban = Goban(board_size, komi=7.5, history_length=history)
        goban.load_game_from_sgf(sgf_file)
        num_moves = len(goban.move_history)
        outcome = goban.ogs_score()
        print("Outcome: ", outcome)
        game_data = {"black_states": goban.full_black_stones_history[:num_moves],
                     "black_liberties": goban.full_black_liberty_history[:num_moves],
                     "white_states": goban.full_white_stones_history[:num_moves],
                     "white_liberties": goban.full_white_liberty_history[:num_moves],
                     "num_moves": num_moves,
                     "moves": goban.move_history,
                     "outcome": outcome}
        return game_data

    def convert(self):
        """
        Run the conversion. On each H5 file group, spin up a new thread to get the data for each SGF file.
        """
        for h5_path, sgf_files in self.h5_file_groups.items():
            with Pool(64) as p:
                data = [(sgf, self.board_size, self.history) for sgf in sgf_files]
                results_queue = p.map(self.load_sgf_into_data, data)

            # Record some useful diagnostic statistics.
            total_black_wins = 0
            total_black_score = 0
            total_white_wins = 0
            total_white_score = 0
            # Build the output file.
            output_file = h5py.File(h5_path, "w")
            games_section = output_file.create_group("games")
            game_number = 0
            # Store all games in this batch to the output file.
            for game_result in results_queue:
                # Extract the score data and add to the stats.
                outcome = game_result["outcome"]
                #print(outcome)
                if outcome.startswith("W"):
                    total_white_wins += 1
                    total_white_score += float(outcome[2:])
                else:
                    total_black_wins += 1
                    total_black_score += float(outcome[2:])
                # Write the game attributes to the file.
                single_game_section = games_section.create_group("game_" + str(game_number))
                single_game_section.create_dataset("outcome", data=game_result["outcome"])
                single_game_section.create_dataset("num_moves", data=game_result["num_moves"], dtype=np.uint8)
                single_game_section.create_dataset("moves", data=game_result["moves"], dtype=np.uint8)
                single_game_section.create_dataset("black_states", data=game_result["black_states"], dtype=np.int8)
                single_game_section.create_dataset("white_states", data=game_result["white_states"], dtype=np.int8)
                single_game_section.create_dataset("black_liberties", data=game_result["black_liberties"], dtype=np.int8)
                single_game_section.create_dataset("white_liberties", data=game_result["white_liberties"], dtype=np.int8)
                game_number += 1
            output_file.close()
            if game_number == 0:
                print("For some reason the game number for this batch is zero so there is nothing to save...")
                continue
            # Print the stats and return the win likelihood of the leading player.
            print(f"On File: {h5_path}")
            print("TW_B: {0}  AWM_B: {1}  TW_W: {2}  AWM_W: {3}".format(total_black_wins,
                                                                        total_black_score / game_number,
                                                                        total_white_wins,
                                                                        total_white_score / game_number))

    @staticmethod
    def get_files_from_dir(data_dir, ext="sgf"):
        """
        Recursively get a list of all the files from the given directory with the given extension.
        :param data_dir:
        :param ext:
        :return:
        """
        file_paths = []
        for current_path, sub_dirs, files in os.walk(data_dir):
            file_paths.extend([os.path.join(current_path, f) for f in files if f.endswith(ext)])
            for sub_dir in sub_dirs:
                file_paths.extend(SGF2H5Writer.get_files_from_dir(sub_dir))
        return file_paths


"""
def save_game_data(game_num, game_outcome, move_history, h5_file):
    This function saves a games history of moves and its outcome into an HDF5 file.
    :param game_num: the integer used to identify this game.
    :param game_outcome: the outcome of the game (1 if black wins, -1 if white wins).
    :param move_history: a list of arrays that holds the state of the board, moving player, and move index.
    :param h5_file: the HDF5 file that data is stored in.
    # Next, create a group for this game.
    game_group = h5_file.create_group("game_" + str(game_num))
    # Create a data set for the games move history.
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
    num_thousands = 0
    game_history_file = h5py.File("downloaded_game_data_" + str(num_thousands) + ".h5", 'w')

    # Load the paths to each game file.
    file_paths = get_files_from_dir("Go_Games_13x13")

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
                board_state.append(np.zeros(169).tolist())
            board_state.append(np.ones(169).tolist())

            # Maintain a history of moves and resulting board states.
            move_history = []

            # Play the game out and store histories.
            for idx in range(0, len(moves), 2):

                # Black moves.
                action_idx = moves[idx]

                # Store the state and action.
                current_board_and_chosen_action = get_single_storable_board_from_state(board_state[14], board_state[15])
                current_board_and_chosen_action.append(1)  # Indicate that the last move was by black.
                current_board_and_chosen_action.append(action_idx)  # Add the move index to this list.
                move_history.append(current_board_and_chosen_action)  # Add this strange data type to the move history list... Why not just use a dictionary?

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
                current_board_and_chosen_action.append(2)  # Indicate that the last move was by white.
                current_board_and_chosen_action.append(action_idx)  # Add the move index.
                move_history.append(current_board_and_chosen_action)  # Again, why not just use a dictionary?

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
            # NOTE: Moves, which range from 0 to 168 should be stored as unsigned bytes (ie dtype='B", not dtype='b')!!!
            move_history = np.array(move_history, dtype='B')

            # Store the outcome of the game.
            if (game_number + 1) % 1000 is 0:
                num_thousands = int((game_number+1) / 1000)
                game_history_file.close()
                game_history_file = h5py.File("downloaded_game_data/downloaded_game_data_" + str(num_thousands) + ".h5", 'w')
            save_game_data(game_number, game_outcome, move_history, game_history_file)

            # Increment the number of games.
            game_number += 1
            if game_number % 100 is 0:
                print("Done with game number " + str(game_number))

        except Exception as e:
            print("Running into error: ", e)
            continue
            
# Run the main code.
load_downloaded_games()

"""


def main():
    # Parse input args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="original_online_games/crap")
    parser.add_argument("--games_per_file", default=2048)
    args = parser.parse_args()

    # Get the SGF files.
    with open("sgf_files.txt", "r") as f:
        sgf_files = [l.strip() for l in f.readlines()]

    # Run the conversion.
    converter = SGF2H5Writer(sgf_files, args.output_dir, args.games_per_file)
    converter.convert()


if __name__ == "__main__":
    main()