"""
This module implements a player controller, which runs on its own thread and processes board states.
It also handles training and updating a go bot.
"""

from nn_ll_tf import PolicyValueNetwork
from threading import Thread, Event
import numpy as np
import logging
import os


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
        self.last_saved_game = ""

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
        self.last_saved_game = output_path
        print("saved to ", self.last_saved_game)

    def get_latest_weights_file(self):
        return self.last_saved_game


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

        # If this player is a follower it will occasionally update its weights to match the leader.
        self.new_weights = self.go_bot_file

        # If this player controller will be responsible for training, store its trainer.
        self.trainer = trainer
        self.bot_name = bot_name

        # Use this lock to prevent issues between threads.
        self.locked = Event()

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
        states_received = len(states)
        num_response_queues = len(response_queues)
        # Convert the newly collected list of states into a numpy batch.
        states_batch = np.asarray(states)
        # Call the policy value network on the batch of states.
        prior_probs, pred_values = self.go_bot.predict_given_state(states_batch, batch_size=self.batch_size)
        # Indicate that processing has started. Sometimes it doesn't, if TF does not clear the GPU...
        # Extract the results and send results back on the response queues.
        num_puts = 0
        for response_queue_id, policy, value in zip(response_queues, prior_probs, pred_values):
            print(response_queue_id)
            response = {"policy": policy, "value": value}
            response_queue = self.output_queue_map[response_queue_id]
            response_queue.put(response)
            num_puts += 1
        print("num response queues: ", num_response_queues, " num states in: ", states_received, " num put outs: ", num_puts)


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
            msg_key = incoming_request["key"]
            msg_data = incoming_request["data"]
            # Make sure the input message is not a shutdown request.
            if msg_key == "SHUTDOWN":
                logging.debug("Processing Queue received the shutdown message. Exiting.")
                return
            # If the message is a training request, start training.
            elif msg_key == "TRAIN":
                self.__train()
                self.locked.set()
                continue
            elif msg_key == "SAVE":
                self.__save()
                self.locked.set()
                continue
            elif msg_key == "UPDATE":
                self.__update(msg_data["new_weights"])
                logging.debug("Updated the weights file to use: " + msg_data["new_weights"])
                self.locked.set()
                continue
            # Otherwise the message is a state that needs to be processed.
            elif msg_key == "STATE":
                print("got state to process")
                states.append(msg_data["state"])
                response_queues.append(msg_data["response_queue_id"])
            # If the number of states to process equals the batch size, process them.
            if len(states) == self.batch_size:
                self.process_states(states, response_queues)
                states = []
                response_queues = []

    def shutdown(self):
        """ When the main script quits, shut down the processing queue."""
        msg = {"key": "SHUTDOWN", "data": None}
        self.input_queue.put(msg)

    def send_save_signal(self):
        """ The the go bot to save its current weights to a file. """
        msg = {"key": "SAVE", "data": None}
        self.input_queue.put(msg)

    def send_train_signal(self):
        """ Tell the go bot to start training. """
        msg = {"key": "TRAIN", "data": None}
        self.input_queue.put(msg)

    def send_update_signal(self, new_weights):
        """ Tell the bot to update its weights given the new weights file (provided by the leader). """
        if new_weights != "":
            msg = {"key": "UPDATE", "data": {"new_weights": new_weights}}
            self.input_queue.put(msg)

    def get_latest_weights_file(self):
        return self.trainer.get_latest_weights_file()

    def __train(self):
        """
        Train the go bot held by this processing queue.
        """
        print("starting training")
        if not self.trainer:
            logging.warning("Attempting to train from PlayerController without a GoBotTrainer.")
        self.trainer.train_bot(self.go_bot)
        print("done training")

    def __save(self):
        """
        Save the go bot's current weights file to the next shodan file.
        """
        print("starting to save...")
        self.trainer.save(self.go_bot)
        print("saved....")

    def __update(self, new_weights):
        """
        Create a new PolicyValue network using the updated weights file.
        """
        self.go_bot.load_model_from_file(new_weights)  # Load the updated wingobot network.

