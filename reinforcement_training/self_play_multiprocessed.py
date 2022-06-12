"""
This script trains a WinGoBot through self play. It saves self play games out to SGF files.

Written By Houston Wingo, Sept 26, 2020

Version Notes:
This version of self play is the most efficient. It uses memoization and true multiprocessing.
Inter-process communication is managed with queues, rather than with pipes.
"""
# Python Standard
import argparse
import multiprocessing as mp
import os
import signal
import sys
from time import time

# Third Party
import h5py
import numpy as np

# Local
sys.path.append("..")
from source.gooop import Goban
from source.training_library import TrainingLibrary
from source.mcts_multiprocess import MonteCarloSearchTree
from source.player_controller import PlayerController, GoBotTrainer

# Globals
KILLED = False
WHITE = "White"
BLACK = "Black"


class SelfPlayGame(mp.Process):
    """
    This class represents a single game of self play between white and black players.
    The game will run on its own thread, but will use a shared queue to add game states for batch processing.
    It stores its game result in the SGF format.
    """

    def __init__(self, game_id, queue_id, black_processing_queues, white_processing_queues, game_results_queue,
                 game_duration, num_leader_simulations, num_follower_simulations, leader, random_seed, komi):
        """
        Initialize a game.
        :param game_id: the unique integer id of this game, used to name the output file.
        :param game_duration: the total number of moves to make in a game.
        :param num_simulations: the number of MCTS simulations to run each move.
        :param black_processing_queues: the input and output queues for the black player.
        :param white_processing_queues: the input and output queues for the white player.
        :param random_seed: a random integer that will be used to seed the numpy random engine on this process.
        """
        # Initialize the superclass.
        mp.Process.__init__(self)

        # Set the numpy random seed in the run function.
        self.random_seed = random_seed

        # Let the first game print info.
        self.can_print = queue_id == 0

        # Store data on this class instance.
        self.game_id = game_id
        self.game_num = queue_id
        self.queue_id = queue_id
        self.output_filename = "self_play_games/test_sgf/game_" + str(game_id) + ".sgf"
        self.black_processing_queues = black_processing_queues
        self.white_processing_queues = white_processing_queues
        self.game_duration = game_duration
        self.num_black_simulations = num_leader_simulations if leader == BLACK else num_follower_simulations
        self.num_white_simulations = num_leader_simulations if leader == WHITE else num_follower_simulations
        self.game_results_queue = game_results_queue

        # Initialize an empty board.
        self.board_state = Goban(13, komi=komi)

        # Create an instance of the MCST for this game, from the perspective of the black and white players.
        self.black_search_tree = MonteCarloSearchTree(self.queue_id, self.black_processing_queues, self.board_state)
        self.white_search_tree = MonteCarloSearchTree(self.queue_id, self.white_processing_queues, self.board_state)

        # The moves made during the game will be stored here, in order. e.g. [B1, W1, B2, W2, ...]
        self.moves = []

        # The winner of the game will be stored in the output file, whenever the game ends.
        self.game_outcome = None

    def play_game(self):
        """ Play a full game. """
        # Make moves until the game duration has been reached. Store these moves.
        for move_number in range(0, self.game_duration, 2):
            # Make a move from the black player's perspective.
            if self.game_num == 0:
                print("Making move {}.".format(move_number))
            best_black_move = self.black_search_tree.search(self.num_black_simulations)
            self.moves.append(best_black_move)
            # Update the white tree to reflect the move. White may have already explored this state.
            if best_black_move in self.white_search_tree.root.children:
                self.white_search_tree.update_root(self.white_search_tree.root.children[best_black_move].copy())
            else:
                curr_board_state = self.black_search_tree.root.get_board_state().copy()
                old_search_count = self.white_search_tree.search_count
                del self.white_search_tree
                self.white_search_tree = MonteCarloSearchTree(self.queue_id,
                                                              self.white_processing_queues,
                                                              curr_board_state,
                                                              search_count=old_search_count)
            # Make a move from the white player's perspective.
            best_white_move = self.white_search_tree.search(self.num_white_simulations)
            self.moves.append(best_white_move)
            # Update the black tree to reflect the move.
            if best_white_move in self.black_search_tree.root.children:
                self.black_search_tree.update_root(self.black_search_tree.root.children[best_white_move].copy())
            else:
                curr_board_state = self.white_search_tree.root.get_board_state().copy()
                old_search_count = self.black_search_tree.search_count
                del self.black_search_tree
                self.black_search_tree = MonteCarloSearchTree(self.queue_id,
                                                              self.black_processing_queues,
                                                              curr_board_state,
                                                              search_count=old_search_count)
        # Now that the game is over, store the final board state. The main thread may want to print it.
        self.board_state = self.black_search_tree.root.get_board_state()  # This state will be from black's perspective.
        # Calculate the winner of the game.
        self.game_outcome = self.calculate_game_outcome()

    def save_game_results_sgf(self):
        """ After a game has been played, the results should be stored to the output file using SGF format. """
        # If the game has not been completed, warn the user that the results cannot be stored yet.
        if not self.game_outcome:
            print("WARNING: Game outcome cannot be saved yet. The game is still in progress.")

        # If the game has completed, convert the list of moves and game outcome to SGF format.
        self.board_state.save_game_to_sgf(self.output_filename)

    def convert_game_result_to_hdf5_format(self):
        """
        Convert the completed game to the HDF5 format used to store game results.
        This function will be called by the main self play thread.
        :return: an entry ready to be stored (by the main thread) in an HDF5 file.
        """
        print("TODO: You need to finish writing this.")

    def calculate_game_outcome(self):
        """ Calculate the outcome of the game, using Tromp Taylor or OGS scoring. """
        #tromp_taylor_result = self.board_state.tromp_taylor_score()
        score_result = self.board_state.ogs_score()
        return score_result

    def store_game_data_on_results_queue(self):
        """
        Call this after game play has completed (self.play_game stores the final Goban to self.board_state).
        """
        num_moves = len(self.board_state.move_history)
        game_data = {"black_states": self.board_state.full_black_stones_history[:num_moves],
                     "black_liberties": self.board_state.full_black_liberty_history[:num_moves],
                     "white_states": self.board_state.full_white_stones_history[:num_moves],
                     "white_liberties": self.board_state.full_white_liberty_history[:num_moves],
                     "num_moves": num_moves,
                     "moves": self.board_state.move_history,
                     "outcome": self.game_outcome}
        self.game_results_queue.put(game_data)

    def run(self):
        """ Run an instance of self play on its own thread. This overwrites the superclass run function. """
        np.random.seed(self.random_seed)
        self.play_game()
        self.store_game_data_on_results_queue()


def write_batch_results_to_h5(filename, results_queue, leader=WHITE):
    """
    This script opens an h5 file, and writes the data held in the results queue to it.
    :param filename:
    :param results_queue:
    :param leader: the color of the leading (training) player for this batch, for statistics reporting purposes.
    :return: the win likelihood for the leading player, from [0.0, 1.0].
    """
    # If the results queue is empty, something has gone wrong...
    if results_queue.empty():
        print("The game results queue is empty... Something has gone wrong for " + filename)
        return 0.0
    # Record some useful diagnostic statistics.
    total_black_wins = 0
    total_black_score = 0
    total_white_wins = 0
    total_white_score = 0
    # Build the output file.
    output_file = h5py.File(filename, "w")
    games_section = output_file.create_group("games")
    game_number = 0
    # Store all games in this batch to the output file.
    while not results_queue.empty():
        game_result = results_queue.get()
        # Extract the score data and add to the stats.
        outcome = game_result["outcome"]
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
        return 0.0
    # Print the stats and return the win likelihood of the leading player.
    print("TW_B: {0}  AWM_B: {1}  TW_W: {2}  AWM_W: {3}".format(
                  total_black_wins, total_black_score / game_number,
                  total_white_wins, total_white_score / game_number))
    if leader == WHITE:
        return total_white_wins / (total_white_wins + total_black_wins)
    else:
        return total_black_wins / (total_black_wins + total_white_wins)


def shutdown(sig, frame):
    global KILLED
    print("should shutdown. got signal")
    print("sig: ", sig)
    print("frame: ", frame)
    KILLED = True


def main():
    """ Run self play on a bunch of threads using this main function. """
    # Parse the input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_threads", default=1, type=int,   # If running just one sim per move, you can probably set this to 2048 or 4096. However, you need to fix a bug in the tree search. Go investigate that later.
                        help="The number of independent games to run in parallel processes.")
    parser.add_argument("--game_duration", default=168, type=int,
                        help="The total number of moves (white and black) to be played each game.")

    parser.add_argument("--num_simulations_leader", default=16, type=int,
                        help="The number of MCTS rollouts for the leading bot to execute.")
    parser.add_argument("--num_simulations_follower", default=2, type=int,
                        help="The number of MCTS rollouts for the following bot to execute.")

    parser.add_argument("--batches_per_training_cycle", default=100, type=int,
                        help="The number of batches to train over, whenever training is executed.")
    parser.add_argument("--training_cycles_per_save", default=100, type=int,
                        help="Save a new weights file after this many training cycles have passed.")

    parser.add_argument("--komi", default=7.5, type=float,
                        help="The bonus points given to white, to account for playing second.")

    parser.add_argument("--weights_dir", default="fresh_weights_06_09_22",
                        help="The directory to save trained weights file to.")
    parser.add_argument("--leading_bot_name", default="wingobot_rl",
                        help="The name to associate the leading bot with.")

    parser.add_argument("--go_bot_1", default="../supervised_training/wingobot_sl_jun_7_2022_hlr/10_0.477.h5",
                        help="The path to the starting weights file of the leading bot.")
    parser.add_argument("--go_bot_2", default="../supervised_training/wingobot_sl_jun_9_2022/1_0.010.h5",
                        help="The path to the starting weights file of the following bot.")

    parser.add_argument("--game_output_dir", default="test_mcts",
                        help="The directory to save self play games, in h5 format, to.")
    args = parser.parse_args()
    num_game_threads = args.game_threads
    game_duration = args.game_duration
    num_leader_simulations = args.num_simulations_leader
    num_follower_simulations = args.num_simulations_follower
    batches_per_training_cycle = args.batches_per_training_cycle
    training_cycles_per_save = args.training_cycles_per_save
    komi = args.komi
    weights_dir = args.weights_dir
    leading_bot_name = args.leading_bot_name
    go_bot_1_file = args.go_bot_1
    go_bot_2_file = args.go_bot_2
    output_dir = args.game_output_dir

    # Define the shutdown function signal handlers.
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Create the multiprocess queues that will store board states (NN inputs) and game results (nparrays for h5 files).
    queue_manager = mp.Manager()
    bot_1_input_queue = queue_manager.Queue()
    bot_2_input_queue = queue_manager.Queue()
    game_results_queue = queue_manager.Queue()

    # Create the output queues and the output queue maps.
    bot_1_output_queues = [queue_manager.Queue() for _ in range(num_game_threads)]
    bot_1_output_queues_map = {idx: output_q for idx, output_q in enumerate(bot_1_output_queues)}
    bot_2_output_queues = [queue_manager.Queue() for _ in range(num_game_threads)]
    bot_2_output_queues_map = {idx: output_q for idx, output_q in enumerate(bot_2_output_queues)}

    # Keep a registry of saved game files.
    game_library = TrainingLibrary(data_dir=output_dir)

    # Create a trainer for the leading player.
    bot_trainer = GoBotTrainer(game_library, weights_dir)

    # Create and start the black and white player processing queues.
    bot_1_processing_queue = PlayerController(go_bot_1_file, num_game_threads, bot_1_input_queue,
                                              bot_name=leading_bot_name,
                                              trainer=bot_trainer)
    bot_1_processing_queue.start()
    bot_2_processing_queue = PlayerController(go_bot_2_file, num_game_threads, bot_2_input_queue)
    bot_2_processing_queue.start()

    # Provide the processing queues with their maps.
    bot_1_processing_queue.set_output_queue_map(bot_1_output_queues_map)
    bot_2_processing_queue.set_output_queue_map(bot_2_output_queues_map)

    # Get the game id offset. TODO make this relative to the number of games historically played.
    game_batch_files = [int(f.split("_")[1].split(".h5")[0]) for f in os.listdir(output_dir) if f.endswith("h5")]
    batch_num = max(game_batch_files) + 1 if game_batch_files else 0
    game_id_offset = batch_num * num_game_threads if game_batch_files else 0

    # Keep track of which color the leading bot plays as during each batch.
    leader = BLACK

    # Use this flag to identify if it is time to swap the followers weights with the leaders.
    recently_saved_weights = False  # Set to true after saving weights, but set to false after updating the follower.

    # Play games until told to quit.
    print("Starting on batch number: %d" % batch_num)
    while not KILLED:

        # Start a timer for this batch, to see how long it runs for.
        start_time = time()

        # Spin up games on their own threads.
        game_threads = []
        for game_num in range(num_game_threads):
            game_id = game_id_offset + game_num
            bot_1_output_queue = bot_1_output_queues[game_num]
            bot_1_queues = {"input": bot_1_input_queue, "output": bot_1_output_queue}
            bot_2_output_queue = bot_2_output_queues[game_num]
            bot_2_queues = {"input": bot_2_input_queue, "output": bot_2_output_queue}
            # On setting random seed: https://discuss.pytorch.org/t/does-getitem-of-dataloader-reset-random-seed/8097/8
            games_random_seed = np.random.randint(0, 4294967296, dtype='uint32')
            # Each batch, alternate between which player is white and which is black.
            # This must be batch_num, not game_num, because each processing queue waits for a full batch.
            if batch_num % 2 == 0:
                # The leading player will play as black.
                leader = BLACK
                new_game = SelfPlayGame(game_id, game_num, bot_1_queues, bot_2_queues, game_results_queue,
                                        game_duration, num_leader_simulations, num_follower_simulations,
                                        leader, games_random_seed, komi=komi)
            else:
                # The leading player will play as white.
                leader = WHITE
                new_game = SelfPlayGame(game_id, game_num, bot_2_queues, bot_1_queues, game_results_queue,
                                        game_duration, num_leader_simulations, num_follower_simulations,
                                        leader, games_random_seed, komi=komi)
            game_threads.append(new_game)

        # Start the games.
        for game in game_threads:
            game.start()

        # Join the running threads when they have completed.
        for completed_game in game_threads:
            completed_game.join()

        # Increment the game id offset.
        game_id_offset += num_game_threads

        # Save the batch data to a single h5 file. Register it with the library.
        batch_output_filename = os.path.join(output_dir, "batch_" + str(batch_num) + ".h5")
        leader_win_likelihood = write_batch_results_to_h5(batch_output_filename, game_results_queue, leader=leader)
        game_library.register_h5_file(batch_output_filename)

        # Notify that we have completed a batch.
        end_time = time()
        total_time = end_time - start_time
        print("Completed batch {0} in {1} seconds.  TGP: {2}  NGT: {3}  SPM: {4}  TGL: {5}  LDR: {6}  LWL: {7}".format(
              batch_num, total_time, int((batch_num + 1) * num_game_threads), num_game_threads,
              num_leader_simulations, game_duration, leader, leader_win_likelihood))
        batch_num += 1

        # If we have now completed a full cycle of batches, train the go bot (only for the leading player).
        if batch_num % batches_per_training_cycle == 0:
            bot_1_processing_queue.send_train_signal()
            bot_1_processing_queue.locked.wait()
            bot_1_processing_queue.locked.clear()

        # If we have completed the requisite number of training cycles, save the weights file for the leading player.
        if batch_num % (batches_per_training_cycle * training_cycles_per_save) == 0:
            bot_1_processing_queue.send_save_signal()
            recently_saved_weights = True
            bot_1_processing_queue.locked.wait()
            bot_1_processing_queue.locked.clear()

        # If we have recently saved the weights and the leader is beating the follower, update the follower.
        if recently_saved_weights and leader_win_likelihood > 0.55:
            leader_weights_path = bot_1_processing_queue.get_latest_weights_file()
            bot_2_processing_queue.send_update_signal(leader_weights_path)
            recently_saved_weights = False
            bot_2_processing_queue.locked.wait()
            bot_2_processing_queue.locked.clear()
        """
        """

    # Join the processing thread.  # TODO Write a proper shutdown handler.
    bot_1_processing_queue.shutdown()
    bot_2_processing_queue.shutdown()
    bot_1_processing_queue.join()
    bot_2_processing_queue.join()


# Run the main function.
if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
