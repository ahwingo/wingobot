"""
This script trains a WinGoBot through self play. It saves self play games out to SGF files.

Written By Houston Wingo, Sept 26, 2020

Version Notes:
This version of self play is the most efficient. It uses memoization and true multiprocessing.
Inter-process communication is managed with queues, rather than with pipes.
"""

import os
import h5py
import random
import argparse
import numpy as np
from time import time
from nn_ll_tf import *
from gooop import Goban
import multiprocessing as mp
from threading import Thread
from mcts_multiprocess import MonteCarloSearchTree
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Sometimes this script hangs. Set this if we actually start processing so we can print a success message, just once.
ITS_WORKING = False


class SelfPlayGame(mp.Process):
    """
    This class represents a single game of self play between white and black players.
    The game will run on its own thread, but will use a shared queue to add game states for batch processing.
    It stores its game result in the SGF format.
    """

    def __init__(self, game_id, black_processing_queues, white_processing_queues, game_results_queue,
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
        self.output_filename = "self_play_games/test_sgf/game_" + str(game_id) + ".sgf"
        self.black_processing_queues = black_processing_queues
        self.white_processing_queues = white_processing_queues
        self.game_duration = game_duration
        self.num_simulations = num_simulations
        self.game_results_queue = game_results_queue

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
        #self.save_game_results_sgf()
        #self.convert_game_result_to_hdf5_format()


class GoBotTrainer:
    """ Instances of this class can be passed to a PlayerController to facilitate the training of its bot. """

    def __init__(self, game_library, weights_directory,
                 train_batch_size=2048,
                 mini_batch_size=32,
                 num_recent_games=1024):
        """
        :param game_library: the TrainingLibrary instance that can be used to train this player on recent games.
        :param train_batch_size: the size of the total batch to train over during a training session.
        :param mini_batch_size: the size of the mini batch to train over. total batches = train_batch_size / mini_bs.
        :param num_recent_games: the number of recent games to pull examples from during training.
        """
        self.game_library = game_library
        self.train_batch_size = train_batch_size
        self.mini_batch_size = mini_batch_size
        self.num_recent_games = num_recent_games
        self.weights_directory = weights_directory
        self.num_shodan_files = self.__get_shodan_file_count()

    def __get_shodan_file_count(self):
        """
        Determine how many shodan files have already been saved.
        """
        shodan_files = [int(f.split("_")[-1].split(".h5")[0])
                        for f in os.listdir(self.weights_directory) if f.endswith("h5")]
        file_count = max(shodan_files) + 1 if shodan_files else 0
        return file_count

    def train_bot(self, bot):
        """
        Using the default training settings of this class, do the following:
            - extract a full batch of training data from the game library
            - call the bot object's training function.
        :param bot: a PolicyValueNetwork object that should be trained.
        """
        game_files = self.game_library.get_last_few_h5_files(self.num_recent_games)
        inputs, policies, values = self.game_library.get_random_training_batch(game_files, self.train_batch_size,
                                                                               bot.history_length, bot.board_size)
        bot.train_supervised(inputs, values, policies, self.mini_batch_size)

    def save(self, bot):
        self.num_shodan_files += 1
        outfile = "shodan_" + bot.name + "_" + str(self.num_shodan_files) + ".h5"
        output_path = os.path.join(self.weights_directory, outfile)
        bot.save_checkpoint(output_path)


class PlayerController(Thread):
    """ Manages the processing of game states across threads. """
    def __init__(self, go_bot_file, batch_size, input_queue, bot_name=None, trainer=None):
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
        self.go_bot = None  # Initialized in the run function.
        self.output_queue_map = None

        # If this player controller will be responsible for training, store its trainer.
        self.trainer = trainer
        self.bot_name = bot_name

    def end_of_batch_cleanup(self):
        self.output_queue_map = None

    def set_output_queue_map(self, output_queue_map):
        """
        :param output_queue_map: a dict matching game ids to their respective queues that the bot should put results in.
        """
        self.output_queue_map = output_queue_map

    def process_states(self, states, response_queues):
        """
        This function runs the policy-value network over a batch of board states and returns the results to the
        response queues of each individual game thread.
        :param states: a list of board states
        :param response_queues: a list of corresponding response queues.
        """
        global ITS_WORKING
        # Convert the newly collected list of states into a numpy batch.
        states_batch = np.asarray(states)
        # Call the policy value network on the batch of states.
        prior_probs, pred_values = self.go_bot.predict_given_state(states_batch, batch_size=self.batch_size)
        # Indicate that processing has started. Sometimes it doesn't, if TF does not clear the GPU...
        if not ITS_WORKING:
            ITS_WORKING = True
            print("Started processing board states.")
        # Extract the results and send results back on the response queues.
        for response_queue_id, policy, value in zip(response_queues, prior_probs, pred_values):
            response = {"policy": policy, "value": value}
            response_queue = self.output_queue_map[response_queue_id]
            response_queue.put(response)

    def run(self):
        """ Process game states until told to stop. """
        self.go_bot = PolicyValueNetwork(0.0001,
                                         train_reinforcement=True,
                                         bot_name=self.bot_name,
                                         starting_network_file=self.go_bot_file)  # Load the wingobot network.
        # Collect states until the batch size is reached. Then process.
        states = []
        response_queues = []
        while True:
            incoming_request = self.input_queue.get()
            # Make sure the input message is not a shutdown request.
            if incoming_request == "SHUTDOWN":
                print("Processing Queue received the shutdown message. Exiting.")
                return
            # If the message is a training request, start training.
            elif incoming_request == "TRAIN":
                self.__train()
                continue
            elif incoming_request == "SAVE":
                self.__save()
                continue
            # Otherwise the message is a state that needs to be processed.
            else:
                states.append(incoming_request["state"])
                response_queues.append(incoming_request["response_queue_id"])
            # If the number of states to process equals the batch size, process them.
            if len(states) == self.batch_size:
                self.process_states(states, response_queues)
                states = []
                response_queues = []

    def shutdown(self):
        """ When the main script quits, shut down the processing queue."""
        self.input_queue.put("SHUTDOWN")

    def send_save_signal(self):
        """ The the go bot to save its current weights to a file. """
        self.input_queue.put("SAVE")

    def send_train_signal(self):
        """ Tell the go bot to start training. """
        self.input_queue.put("TRAIN")

    def __train(self):
        """ 
        Train the go bot held by this processing queue.
        """
        if not self.trainer:
            print("WARNING: Attempting to train from PlayerController without a GoBotTrainer.")
        self.trainer.train_bot(self.go_bot)

    def __save(self):
        """
        Save the go bot's current weights file to the next shodan file.
        """
        self.trainer.save(self.go_bot)


class TrainingLibrary:
    """
    This class is used to keep track of self play game h5 files.
    It stores path, and on request, will present:
        - a list of files to pull training data from
        - a minibatch of training data
    """
    def __init__(self):
        self.registered_h5_files = []

    def register_h5_file(self, h5_file):
        """
        Add a self play games file to the list.
        :param h5_file:
        :return:
        """
        self.registered_h5_files.append(h5_file)

    def get_last_few_h5_files(self, last_few):
        """
        Return the last few self play files that were registered.
        :param last_few: an integer number of the last few files to return.
        :return: a list of file paths.
        """
        if last_few > len(self.registered_h5_files):
            return self.registered_h5_files
        return self.registered_h5_files[-last_few:]

    @staticmethod
    def get_num_pads(move_number, history_length):
        """
        Helper function to find the number of pads to use for the given history length and move number.
        :param move_number:
        :param history_length:
        :return:
        """
        return abs(min(0, move_number - history_length))

    @staticmethod
    def get_actual_states_range(move_number, history_length):
        """
        Helper function to find the range of indices to pull actual board states from.
        :param move_number:
        :param history_length:
        :return:
        """
        return range(max(0, move_number - history_length), move_number, 1)

    @staticmethod
    def get_random_training_batch(h5_files, batch_size, history_size, board_size):
        """
        Return a randomly selected batch of board states and the ground truth policy and values.
        :param h5_files: a list of h5 files to train over.
        :param batch_size:
        :param history_size:
        :param board_size:
        :return:
        """
        open_game_files = [h5py.File(h5f, "r") for h5f in h5_files]
        inputs = []
        gt_values = []
        gt_policies = []
        for _ in range(batch_size):
            random_game_file = random.choice(open_game_files)
            random_game_num = random.choice(list(random_game_file["games"].keys()))
            random_game = random_game_file["games"][random_game_num]
            num_moves = random_game["num_moves"][()]
            random_move = random.randint(0, num_moves - 1)

            # Build the input.
            input_layers = []
            num_pads = 2 * TrainingLibrary.get_num_pads(random_move, history_size)  # 2X b/c of black & white layers.
            for _ in range(num_pads):
                input_layers.append(np.zeros((board_size, board_size), dtype=np.int8))
            for idx in TrainingLibrary.get_actual_states_range(random_move, history_size):
                if random_move % 2 == 0:  # its black to move
                    input_layers.append(random_game["black_states"][()][idx])
                    input_layers.append(random_game["white_states"][()][idx])
                else:  # its white to move
                    input_layers.append(random_game["white_states"][()][idx])
                    input_layers.append(random_game["black_states"][()][idx])
            if random_move % 2 == 0:
                input_layers.append(np.ones((board_size, board_size), dtype=np.int8))  # player identity layer: B=1 W=0
                if random_move == 0:
                    input_layers.append(np.zeros((board_size, board_size), dtype=np.int8))
                    input_layers.append(np.zeros((board_size, board_size), dtype=np.int8))
                else:
                    input_layers.append(random_game["black_liberties"][()][random_move - 1])
                    input_layers.append(random_game["white_liberties"][()][random_move - 1])
            else:
                input_layers.append(np.zeros((board_size, board_size), dtype=np.int8))  # player identity layer: B=1 W=0
                if random_move == 0:
                    input_layers.append(np.zeros((board_size, board_size), dtype=np.int8))
                    input_layers.append(np.zeros((board_size, board_size), dtype=np.int8))
                else:
                    input_layers.append(random_game["white_liberties"][()][random_move - 1])
                    input_layers.append(random_game["black_liberties"][()][random_move - 1])

            # Extract the policy.
            policy = np.zeros(1 + board_size ** 2, dtype=np.int8)
            row, col = random_game["moves"][()][random_move]
            move_idx = min(row * board_size + col, 169)  # 169 indicates a pass (row = col = 13  --->  182)
            policy[move_idx] = 1

            # Extract the value.
            outcome = random_game["outcome"][()]
            if random_move % 2 == 0:
                value = 1 if outcome[0] == "B" else -1
            else:
                value = 1 if outcome[0] == "W" else -1

            # Add the input, policy, and value to the list.
            inputs.append(input_layers)
            gt_policies.append(policy)
            gt_values.append(value)

        # Close any files that you opened.
        [f.close() for f in open_game_files]
        # Return the inputs, policies, and values.
        inputs = np.asarray(inputs)
        gt_values = np.reshape(np.asarray(gt_values), (batch_size, 1))
        gt_policies = np.asarray(gt_policies)
        return inputs, gt_policies, gt_values


def write_batch_results_to_h5(filename, results_queue):
    """
    This script opens an h5 file, and writes the data held in the results queue to it.
    :param filename:
    :param results_queue:
    :return:
    """
    output_file = h5py.File(filename, "w")
    games_section = output_file.create_group("games")
    game_number = 0
    while not results_queue.empty():
        game_result = results_queue.get()
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


def main():
    """ Run self play on a bunch of threads using this main function. """
    # Parse the input arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_threads", default=12, type=int)
    parser.add_argument("--game_duration", default=15, type=int)
    parser.add_argument("--num_simulations", default=5, type=int)
    parser.add_argument("--batches_per_training_cycle", default=1, type=int)
    parser.add_argument("--training_cycles_per_save", default=4, type=int)
    parser.add_argument("--weights_dir", default="shodan_dark_rock")
    parser.add_argument("--leading_bot_name", default="dark_rock")
    parser.add_argument("--go_bot_1", default="young_dark_rock/young_dark_rock_ckpt_11500.h5")
    parser.add_argument("--go_bot_2", default="young_dark_rock/young_dark_rock_ckpt_12000.h5")
    parser.add_argument("--game_output_dir", default="self_play_games/h5_games")
    args = parser.parse_args()
    num_game_threads = args.game_threads
    game_duration = args.game_duration
    num_simulations = args.num_simulations
    batches_per_training_cycle = args.batches_per_training_cycle
    training_cycles_per_save = args.training_cycles_per_save
    weights_dir = args.weights_dir
    leading_bot_name = args.leading_bot_name
    go_bot_1_file = args.go_bot_1
    go_bot_2_file = args.go_bot_2
    output_dir = args.game_output_dir

    # Create the multiprocess queues that will store board states (NN inputs) and game results (nparrays for h5 files).
    queue_manager = mp.Manager()
    bot_1_input_queue = queue_manager.Queue()
    bot_2_input_queue = queue_manager.Queue()
    game_results_queue = queue_manager.Queue()

    # Keep a registry of saved game files.
    game_library = TrainingLibrary()

    # Create a trainer for the leading player.
    bot_trainer = GoBotTrainer(game_library, weights_dir)

    # Create and start the black and white player processing queues.
    bot_1_processing_queue = PlayerController(go_bot_1_file, num_game_threads, bot_1_input_queue,
                                              bot_name=leading_bot_name,
                                              trainer=bot_trainer)
    bot_1_processing_queue.start()
    bot_2_processing_queue = PlayerController(go_bot_2_file, num_game_threads, bot_2_input_queue)
    bot_2_processing_queue.start()

    # Get the game id offset. TODO make this relative to the number of games historically played.
    game_batch_files = [int(f.split("_")[1].split(".h5")[0]) for f in os.listdir(output_dir) if f.endswith("h5")]
    batch_num = max(game_batch_files) + 1 if game_batch_files else 0
    num_weights_file_updates = 0  # TODO Give better name... Make relative to the number of shodan files...
    game_id_offset = batch_num * num_game_threads if game_batch_files else 0

    # Play games until told to quit.
    print("Starting on batch number: ", batch_num)
    while True:

        # Start a timer for this batch, to see how long it runs for.
        start_time = time()

        # Spin up games on their own threads.
        bot_1_output_queue_map = {}
        bot_2_output_queue_map = {}
        game_threads = []
        for game_num in range(num_game_threads):
            game_id = game_id_offset + game_num
            bot_1_output_queue = queue_manager.Queue()
            bot_1_output_queue_map[game_id] = bot_1_output_queue
            bot_1_queues = {"input": bot_1_input_queue, "output": bot_1_output_queue}
            bot_2_output_queue = queue_manager.Queue()
            bot_2_output_queue_map[game_id] = bot_2_output_queue
            bot_2_queues = {"input": bot_2_input_queue, "output": bot_2_output_queue}
            # On setting random seed: https://discuss.pytorch.org/t/does-getitem-of-dataloader-reset-random-seed/8097/8
            games_random_seed = np.random.randint(0, 4294967296, dtype='uint32')
            # Each batch, alternate between which player is white and which is black.
            # This must be batch_num, not game_num, because each processing queue waits for a full batch.
            if batch_num % 2 == 0:
                new_game = SelfPlayGame(game_id, bot_1_queues, bot_2_queues, game_results_queue,
                                        game_duration, num_simulations, games_random_seed)
            else:
                new_game = SelfPlayGame(game_id, bot_2_queues, bot_1_queues, game_results_queue,
                                        game_duration, num_simulations, games_random_seed)
            game_threads.append(new_game)

        # Provide the processing queues with their maps.
        bot_1_processing_queue.set_output_queue_map(bot_1_output_queue_map)
        bot_2_processing_queue.set_output_queue_map(bot_2_output_queue_map)

        # Start the games.
        for game in game_threads:
            game.start()

        # Join the running threads when they have completed.
        for completed_game in game_threads:
            completed_game.join()

        # Increment the game id offset.
        game_id_offset += num_game_threads

        # Clean up the black and white processing queues.
        bot_1_processing_queue.end_of_batch_cleanup()
        bot_2_processing_queue.end_of_batch_cleanup()

        # Save the batch data to a single h5 file. Register it with the library.
        batch_output_filename = os.path.join(output_dir, "batch_" + str(batch_num) + ".h5")
        write_batch_results_to_h5(batch_output_filename, game_results_queue)
        game_library.register_h5_file(batch_output_filename)

        # Notify that we have completed a batch.
        end_time = time()
        total_time = end_time - start_time
        print("Completed batch ", batch_num, " in ", total_time, " seconds. ",
              "TGP: ", int((batch_num + 1) * num_game_threads), " NGT: ", num_game_threads,
              " SPG: ", num_simulations, " TGL: ", game_duration)
        batch_num += 1

        # If we have now completed a full cycle of batches, train the go bot (only for the leading player).
        if batch_num % batches_per_training_cycle == 0:
            bot_1_processing_queue.send_train_signal()
            num_weights_file_updates += 1

        # If we have completed the requisite number of training cycles, save the weights file for the leading player.
        if num_weights_file_updates % training_cycles_per_save == 0:
            bot_1_processing_queue.send_save_signal()

    # Join the processing thread.  # TODO Write a proper shutdown handler.
    bot_1_processing_queue.shutdown()
    bot_2_processing_queue.shutdown()
    bot_1_processing_queue.join()
    bot_2_processing_queue.join()


# Run the main function.
if __name__ == "__main__":
    main()
