"""
This module is used to keep track of self play game files.
"""
import h5py
import random
import numpy as np


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
