"""
This script trains a WinGoBot through self play. It saves self play games out to SGF files.

Written By Houston Wingo, Sept 26, 2020

Version Notes:
This version of self play is the most efficient. It uses memoization and true multiprocessing.
Inter-process communication is managed with queues, rather than with pipes.
"""

import os
import sys
import h5py
import random
import signal
import argparse
import numpy as np
from time import time
from gooop import Goban
from threading import Event
import multiprocessing as mp
from training_library import TrainingLibrary
from mcts_multiprocess import MonteCarloSearchTree
from player_controller import PlayerController, GoBotTrainer
from threading import Thread


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

        # Store data on this class instance.
        self.game_id = game_id
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
        self.original_search_tree = self.black_search_tree

        # The moves made during the game will be stored here, in order. e.g. [B1, W1, B2, W2, ...]
        self.moves = []

        # The winner of the game will be stored in the output file, whenever the game ends.
        self.game_outcome = None

    def __sizeof__(self):
        """
        Get the full size of this object and its references.
        """
        total_size = 0
        total_size += sys.getsizeof(self.random_seed)
        total_size += sys.getsizeof(self.game_id)
        total_size += sys.getsizeof(self.queue_id)
        total_size += sys.getsizeof(self.output_filename)
        total_size += sys.getsizeof(self.black_processing_queues)
        total_size += sys.getsizeof(self.white_processing_queues)
        total_size += sys.getsizeof(self.game_duration)
        total_size += sys.getsizeof(self.num_black_simulations)
        total_size += sys.getsizeof(self.num_white_simulations)
        total_size += sys.getsizeof(self.game_results_queue)
        total_size += sys.getsizeof(self.board_state)
        total_size += sys.getsizeof(self.black_search_tree)
        total_size += sys.getsizeof(self.white_search_tree)
        total_size += sys.getsizeof(self.moves)
        total_size += sys.getsizeof(self.game_outcome)
        return total_size

    def play_game(self):
        """ Play a full game. """
        # Make moves until the game duration has been reached. Store these moves.
        for move_number in range(0, self.game_duration, 2):
            total_size = sys.getsizeof(self)
            # Make a move from the black player's perspective.
            best_black_move = self.black_search_tree.search(self.num_black_simulations)
            self.moves.append(best_black_move)
            # Update the white tree to reflect the move. White may have already explored this state.
            """
            if best_black_move in self.white_search_tree.root.children:
                self.white_search_tree.update_root(self.white_search_tree.root.children[best_black_move])
            else:
                curr_board_state = self.black_search_tree.root.get_board_state()
                self.white_search_tree = MonteCarloSearchTree(self.queue_id,
                                                              self.white_processing_queues,
                                                              curr_board_state)
            """
            curr_board_state = self.black_search_tree.root.get_board_state().copy()
            self.white_search_tree.update_on_opponent_selection(best_black_move, curr_board_state)

            # Make a move from the white player's perspective.
            best_white_move = self.white_search_tree.search(self.num_white_simulations)
            self.moves.append(best_white_move)
            """
            # Update the black tree to reflect the move.
            if best_white_move in self.black_search_tree.root.children:
                self.black_search_tree.update_root(self.black_search_tree.root.children[best_white_move])
            else:
                curr_board_state = self.white_search_tree.root.get_board_state()
                self.black_search_tree = MonteCarloSearchTree(self.queue_id,
                                                              self.black_processing_queues,
                                                              curr_board_state)
            """
            curr_board_state = self.white_search_tree.root.get_board_state().copy()
            self.black_search_tree.update_on_opponent_selection(best_white_move, curr_board_state)
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
        print("\n\n=============================================================================================\n")
        print(self.original_search_tree.print_tree("tree_print_test.sgf"))


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


class FakeGoBot(Thread):
    """
    Instances of this class will randomly select a move and stick it back on the results queue.
    To be used for speed testing of the self play MCTS algorithm.
    """
    def __init__(self, batch_size, input_queue):
        """
        Construct an instance of the FakeGoBot. Similar to the PlayerController.
        """
        # Initialize the superclass.
        Thread.__init__(self)
        # Set instance variables.
        self.input_queue = input_queue
        self.batch_size = batch_size
        self.output_queue_map = None

        # Use this lock to prevent issues between threads.
        self.locked = Event()

    def end_of_batch_cleanup(self):
        self.output_queue_map = None

    def set_output_queue_map(self, output_queue_map):
        """
        :param output_queue_map: a dict matching game ids to their respective queues that the bot should put results in.
        """
        self.output_queue_map = output_queue_map

    def process_states(self, response_queues):
        """
        This function runs the policy-value network over a batch of board states and returns the results to the
        response queues of each individual game thread.
        :param states: a list of board states
        :param response_queues: a list of corresponding response queues.
        """
        # Extract the results and send results back on the response queues.
        num_puts = 0
        policy = np.random.random(170)
        value = np.random.random(1)
        for response_queue_id in response_queues:
            response = {"policy": policy, "value": value}
            response_queue = self.output_queue_map[response_queue_id]
            response_queue.put(response)
            num_puts += 1

    def run(self):
        """ Process game states until told to stop. """
        # Collect states until the batch size is reached. Then process.
        states = []
        response_queues = []
        while True:
            incoming_request = self.input_queue.get()
            msg_key = incoming_request["key"]
            msg_data = incoming_request["data"]
            # Make sure the input message is not a shutdown request.
            if msg_key == "SHUTDOWN":
                print("Processing Queue received the shutdown message. Exiting.")
                return
            elif msg_key == "STATE":
                states.append(msg_data["state"])
                response_queues.append(msg_data["response_queue_id"])
            # If the number of states to process equals the batch size, process them.
            if len(states) == self.batch_size:
                self.process_states(response_queues)
                states = []
                response_queues = []

    def shutdown(self):
        """ When the main script quits, shut down the processing queue."""
        msg = {"key": "SHUTDOWN", "data": None}
        self.input_queue.put(msg)


def speed_test(num_game_threads, game_duration, num_leader_simulations, num_follower_simulations, komi, trials, real_bot=False):
    # Define the shutdown signals.
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

    # Create and start the black and white player processing queues.
    if real_bot:
        bot_1_processing_queue = PlayerController("shodan_fossa/shodan_focal_fossa_101.h5", num_game_threads, bot_1_input_queue)
        bot_1_processing_queue.start()
        bot_2_processing_queue = PlayerController("shodan_fossa/shodan_focal_fossa_102.h5", num_game_threads, bot_2_input_queue)
        bot_2_processing_queue.start()
    else:
        bot_1_processing_queue = FakeGoBot(num_game_threads, bot_1_input_queue)
        bot_1_processing_queue.start()
        bot_2_processing_queue = FakeGoBot(num_game_threads, bot_2_input_queue)
        bot_2_processing_queue.start()

    # Provide the processing queues with their maps.
    bot_1_processing_queue.set_output_queue_map(bot_1_output_queues_map)
    bot_2_processing_queue.set_output_queue_map(bot_2_output_queues_map)

    # Keep track of which color the leading bot plays as during each batch.
    leader = BLACK

    # Play games until told to quit.
    for trial_num in range(trials):

        # Start a timer for this batch, to see how long it runs for.
        start_time = time()

        # Spin up games on their own threads.
        game_threads = []
        for game_num in range(num_game_threads):
            game_id = game_num
            bot_1_output_queue = bot_1_output_queues[game_num]
            bot_1_queues = {"input": bot_1_input_queue, "output": bot_1_output_queue}
            bot_2_output_queue = bot_2_output_queues[game_num]
            bot_2_queues = {"input": bot_2_input_queue, "output": bot_2_output_queue}
            # On setting random seed: https://discuss.pytorch.org/t/does-getitem-of-dataloader-reset-random-seed/8097/8
            games_random_seed = np.random.randint(0, 4294967296, dtype='uint32')
            # Each batch, alternate between which player is white and which is black.
            # This must be batch_num, not game_num, because each processing queue waits for a full batch.
            if trial_num % 2 == 0:
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

        # Notify that we have completed a batch.
        end_time = time()
        total_time = end_time - start_time
        total_games_played = int((trial_num + 1) * num_game_threads)
        total_states_evaluated = num_game_threads*(num_leader_simulations+num_follower_simulations)*game_duration/2
        states_per_second = total_states_evaluated / total_time
        print("Comp. batch {0} in {1} secs. TGP: {2} NGT: {3} SPM {4} TGL: {5} SPS: {6}".format(trial_num,
                                                                                                total_time,
                                                                                                total_games_played,
                                                                                                num_game_threads,
                                                                                                num_leader_simulations,
                                                                                                game_duration,
                                                                                                states_per_second))

    # Join the processing thread.
    bot_1_processing_queue.shutdown()
    bot_2_processing_queue.shutdown()
    bot_1_processing_queue.join()
    bot_2_processing_queue.join()


def main():
    """ Run self play on a bunch of threads using this main function. """
    # Parse the input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--game_threads", default=1, type=int)
    parser.add_argument("-gd", "--game_duration", default=32, type=int)
    parser.add_argument("-nsl", "--num_simulations_leader", default=200, type=int)
    parser.add_argument("-nsf", "--num_simulations_follower", default=2, type=int)
    parser.add_argument("--komi", default=6.5, type=float)
    parser.add_argument("--trials", default=1, type=float, help="Number of speed tests to run and average over.")
    args = parser.parse_args()
    num_game_threads = args.game_threads
    game_duration = args.game_duration
    num_leader_simulations = args.num_simulations_leader
    num_follower_simulations = args.num_simulations_follower
    komi = args.komi
    trials = args.trials
    speed_test(num_game_threads, game_duration, num_leader_simulations, num_follower_simulations, komi, trials, real_bot=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()


