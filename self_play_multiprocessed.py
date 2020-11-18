"""
This script trains a WinGoBot through self play. It saves self play games out to SGF files.

Written By Houston Wingo, Sept 26, 2020

Version Notes:
This version of self play is the most efficient. It uses memoization and true multiprocessing.
Inter-process communication is managed with queues, rather than with pipes.
"""

import os
import argparse
import numpy as np
from time import time
from gooop import Goban
import multiprocessing as mp
from threading import Thread
from mcts_multiprocess import MonteCarloSearchTree
from nn_ll_tf import *
#from nn_ll_tf import PolicyValueNetwork
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

ITS_WORKING = False


class SelfPlayGame(mp.Process):
    """
    This class represents a single game of self play between white and black players.
    The game will run on its own thread, but will use a shared queue to add game states for batch processing.
    It stores its game result in the SGF format.
    """

    def __init__(self, game_id, black_processing_queues, white_processing_queues,
                 game_duration, num_simulations, random_seed):
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
        self.output_filename = "self_play_games/sgf/game_" + str(game_id) + ".sgf"
        self.black_processing_queues = black_processing_queues
        self.white_processing_queues = white_processing_queues
        self.game_duration = game_duration
        self.num_simulations = num_simulations

        # Initialize an empty board.
        self.board_state = Goban(13)

        # Create an instance of the MCST for this game, from the perspective of the black and white players.
        self.black_search_tree = MonteCarloSearchTree(self.game_id, self.black_processing_queues, self.board_state)
        self.white_search_tree = MonteCarloSearchTree(self.game_id, self.white_processing_queues, self.board_state)

        # The moves made during the game will be stored here, in order. e.g. [B1, W1, B2, W2, ...]
        self.moves = []

        # The winner of the game will be stored in the output file, whenever the game ends.
        self.game_outcome = None

    def play_game(self):
        """ Play a full game. """
        # Make moves until the game duration has been reached. Store these moves.
        for move_number in range(0, self.game_duration, 2):

            # Make a move from the black player's perspective.
            best_black_move = self.black_search_tree.search(self.num_simulations)
            self.moves.append(best_black_move)
            # Update the white tree to reflect the move. White may have already explored this state.
            if best_black_move in self.white_search_tree.root.children:
                self.white_search_tree.update_root(self.white_search_tree.root.children[best_black_move])
            else:
                curr_board_state = self.black_search_tree.root.get_board_state()
                self.white_search_tree = MonteCarloSearchTree(self.game_id,
                                                              self.white_processing_queues,
                                                              curr_board_state)

            # Make a move from the white player's perspective.
            best_white_move = self.white_search_tree.search(self.num_simulations)
            self.moves.append(best_white_move)
            # Update the black tree to reflect the move.
            if best_white_move in self.black_search_tree.root.children:
                self.black_search_tree.update_root(self.black_search_tree.root.children[best_white_move])
            else:
                curr_board_state = self.white_search_tree.root.get_board_state()
                self.black_search_tree = MonteCarloSearchTree(self.game_id,
                                                              self.black_processing_queues,
                                                              curr_board_state)

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
        """ Calculate the outcome of the game, using Tromp Taylor scoring. """
        # Calculate the Tromp Taylor score.
        tromp_taylor_result = self.board_state.tromp_taylor_score()
        return tromp_taylor_result

    def run(self):
        """ Run an instance of self play on its own thread. This overwrites the superclass run function. """
        np.random.seed(self.random_seed)
        self.play_game()
        self.save_game_results_sgf()
        #self.convert_game_result_to_hdf5_format()


#class ProcessingQueue(mp.Process):
class ProcessingQueue(Thread):
    """ Manages the processing of game states across threads. """

    def __init__(self, go_bot_file, batch_size, input_queue):
        """
        Initialize an instance of the processing queue, on the main thread.
        :param go_bot_file: the weights file of the neural network that will process game states.
        :param batch_size: the number of game states to process all at once.
        :param input_queue: the queue that the game subprocesses will put states in.
        """
        # Initialize the superclass.
        Thread.__init__(self)
        # Set instance variables.
        self.input_queue = input_queue
        self.batch_size = batch_size
        self.go_bot_file = go_bot_file
        self.call_count = 0  # The number of times this processing queue has been called.
        self.go_bot = None  # Initialized in the run function.
        self.output_queue_map = None

    def end_of_batch_cleanup(self):
        self.output_queue_map = None

    def set_output_queue_map(self, output_queue_map):
        """
        :param output_queue_map: a dict matching game ids to their respective queues that the bot should put results in.
        """
        self.output_queue_map = output_queue_map

    def run(self):
        """ Process game states until told to stop. """
        global ITS_WORKING
        self.go_bot = PolicyValueNetwork(0.0001, starting_network_file=self.go_bot_file)  # Load the wingobot network.
        while True:
            # Collect states until the batch size is reached. Then process.
            states = []
            response_queues = []
            for _ in range(self.batch_size):
                incoming_request = self.input_queue.get()
                # Make sure the input message is not a shutdown request.
                if incoming_request == "SHUTDOWN":
                    print("Processing Queue received the shutdown message. Exiting.")
                    return
                states.append(incoming_request["state"])
                response_queues.append(incoming_request["response_queue_id"])
            # Convert the newly collected list of states into a numpy batch.
            states_batch = np.asarray(states)
            # Call the policy value network on the batch of states.
            prior_probs, pred_values = self.go_bot.predict_given_state(states_batch, batch_size=self.batch_size)

            # Print the number of times this has run.
            # print("Num Processing Calls: ", self.call_count)
            self.call_count += 1
            if not ITS_WORKING:
                ITS_WORKING = True
                print("Started processing board states.")

            # Extract the results and send results back on the response queues.
            for response_queue_id, policy, value in zip(response_queues, prior_probs, pred_values):
                response = {"policy": policy, "value": value}
                response_queue = self.output_queue_map[response_queue_id]
                response_queue.put(response)

    def shutdown(self):
        """ When the main script quits, shut down the processing queue."""
        self.input_queue.put("SHUTDOWN")


def main():
    """ Run self play on a bunch of threads using this main function. """
    # Parse the input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_threads", default=128, type=int)
    parser.add_argument("--game_duration", default=125, type=int)
    parser.add_argument("--num_simulations", default=5, type=int)
    #parser.add_argument("--black_go_bot", default="young_thread_ripper_ckpt_11500.h5")
    parser.add_argument("--black_go_bot", default="young_dark_rock_ckpt_11500.h5")
    parser.add_argument("--white_go_bot", default="young_dark_rock_ckpt_12000.h5")
    args = parser.parse_args()
    num_game_threads = args.game_threads
    game_duration = args.game_duration
    num_simulations = args.num_simulations
    black_go_bot_file = args.black_go_bot
    white_go_bot_file = args.white_go_bot

    # Create the processing queue and start it.
    queue_manager = mp.Manager()
    black_input_queue = queue_manager.Queue()
    white_input_queue = queue_manager.Queue()

    # Create and start the black and white player processing queues.
    black_processing_queue = ProcessingQueue(black_go_bot_file, num_game_threads, black_input_queue)
    black_processing_queue.start()
    white_processing_queue = ProcessingQueue(white_go_bot_file, num_game_threads, white_input_queue)
    white_processing_queue.start()

    # Get the game id offset. TODO make this relative to the number of games historically played.
    game_id_offset = 0

    # Play games until told to quit.
    # for batch_num in range(5):
    batch_num = 0
    while True:

        # Start a timer for this batch, to see how long it runs for.
        start_time = time()

        # Spin up games on their own threads.
        black_output_queue_map = {}
        white_output_queue_map = {}
        game_threads = []
        for game_num in range(num_game_threads):
            game_id = game_id_offset + game_num
            black_output_queue = queue_manager.Queue()
            black_output_queue_map[game_id] = black_output_queue
            black_queues = {"input": black_input_queue, "output": black_output_queue}
            white_output_queue = queue_manager.Queue()
            white_output_queue_map[game_id] = white_output_queue
            white_queues = {"input": white_input_queue, "output": white_output_queue}
            # On setting random seed: https://discuss.pytorch.org/t/does-getitem-of-dataloader-reset-random-seed/8097/8
            games_random_seed = np.random.randint(0, 4294967296, dtype='uint32')
            new_game = SelfPlayGame(game_id, black_queues, white_queues,
                                    game_duration, num_simulations, games_random_seed)
            game_threads.append(new_game)

        # Provide the processing queues with their maps.
        black_processing_queue.set_output_queue_map(black_output_queue_map)
        white_processing_queue.set_output_queue_map(white_output_queue_map)

        # Start the games.
        for game in game_threads:
            game.start()

        # Join the running threads when they have completed.
        for completed_game in game_threads:
            completed_game.join()

        # Increment the game id offset.
        game_id_offset += num_game_threads

        # Clean up the black and white processing queues.
        black_processing_queue.end_of_batch_cleanup()
        white_processing_queue.end_of_batch_cleanup()

        # Notify that we have completed a batch.
        end_time = time()
        total_time = end_time - start_time
        print("Completed batch ", batch_num, " in ", total_time, " seconds. ",
              "NGT: ", num_game_threads, " SPG: ", num_simulations, " TGL: ", game_duration)
        batch_num += 1

    # Join the processing thread.
    black_processing_queue.shutdown()
    white_processing_queue.shutdown()
    black_processing_queue.join()
    white_processing_queue.join()

# Run the main function.
if __name__ == "__main__":
    main()
