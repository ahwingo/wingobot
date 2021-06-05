"""
This module implements a Goban class and functions related to storing and scoring Go games.
"""

import numpy as np
import random
import copy
import sys
sys.path.append("score-estimator")
from ogs_estimator import estimate as ogs_estimate


class String:
    """
    This class may be used to represent a string and its liberty counts. Maybe.
    """
    def __init__(self, string_id, indices, liberties):
        self.string_id = string_id
        self.indices = indices
        self.liberties = liberties

    def __sizeof__(self):
        """
        Get the total size of this object and its attributes.
        """
        total_size = 0
        total_size += sys.getsizeof(self.string_id)
        total_size += sys.getsizeof(self.indices)
        total_size += sys.getsizeof(self.liberties)
        return total_size

    def liberty_count(self):
        return len(self.liberties)

    def remove_liberty(self, lib_row, lib_col):
        """
        Remove a liberty from the string, if the liberty index is part of this string.
        :param lib_row:
        :param lib_col:
        """
        if [lib_row, lib_col] in self.liberties:
            self.liberties.remove([lib_row, lib_col])

    def add_liberty(self, lib_row, lib_col):
        """
        Add a liberty to the string, if it is not already included.
        :param lib_row:
        :param lib_col:
        :return:
        """
        if [lib_row, lib_col] not in self.liberties:
            self.liberties.append([lib_row, lib_col])


class Goban:
    """
    Instances of this class represent a Goban (Go board) and its history.
    """
    def __init__(self, size, komi=7.5, history_length=8):
        """
        Create an instance of a Goban and its various nparrays that will be used to store the board state.
        :param size: the size of a side of the square board (eg. 9, 13, 19).
        :param history_length: the number of historical moves to store.
        """
        # Use these values to represent the white and black players.
        self.black = 1
        self.white = -1
        self.no_move = size**2  # Represents a pass.

        # Store the user provided params.
        self.size = size
        self.history_length = history_length

        # Store the move history (indices of the move). Even indices for a black move, odd for white.
        self.move_history = []
        self.pass_idx = [size, size]

        # Create an empty board to represent the stored history of the game. This is part of what is passed to the bot.
        self.full_history = np.zeros((2*history_length, size, size))

        # Store all states that will eventually be saved to / reloaded from an h5 file.
        self.max_moves = size**2 + size  # TODO You may need to raise this threshold. It appears there is an odd bug causing an issue related to this.
        self.full_black_stones_history = np.zeros((self.max_moves, size, size), dtype=np.int8)
        self.full_black_liberty_history = np.zeros((self.max_moves, size, size), dtype=np.int8)
        self.full_white_stones_history = np.zeros((self.max_moves, size, size), dtype=np.int8)
        self.full_white_liberty_history = np.zeros((self.max_moves, size, size), dtype=np.int8)

        # Create a single printable board, which can also be used to calculate the score.
        # Also create one for the previous state.
        self.previous_board = np.zeros((size, size))
        self.current_board = np.zeros((size, size))
        self.komi = komi

        # Keep track of the active black and white strings.
        self.black_string_count = 0  # Strings will be referenced as "B0", "B1", etc.
        self.white_string_count = 0  # Strings will be referenced as "W0", "W1", etc.
        self.strings = {}  # Uses the string ids to keep track of strings.
        self.string_board = np.reshape(np.asarray(["EMPTY"] * (self.size**2)), (self.size, self.size))

        # Create matrices to represent the player to move next (1 == black, 0 == white).
        self.current_player = self.black
        self.black_to_go_next = np.ones((size, size))
        self.white_to_go_next = np.zeros((size, size))

        # Create two matrices to represent the black and white liberties.
        self.black_liberties = np.zeros((size, size))
        self.white_liberties = np.zeros((size, size))

        # Also keep track of the liberties available at each intersection.
        self.available_liberties = np.ones((size, size)) * 4
        self.available_liberties[0, :] -= 1
        self.available_liberties[size - 1, :] -= 1
        self.available_liberties[:, 0] -= 1
        self.available_liberties[:, size - 1] -= 1

        # Create matrices to represent the legal moves available to the white and black players.
        self.legal_black_moves = np.ones((size, size))
        self.legal_white_moves = np.ones((size, size))

        # Store the position that is illegal for the next move based on the Ko rule (be sure to clear after move made).
        self.illegal_pos_by_ko = None

    def __sizeof__(self):
        """
        Get the total size of this object and all of its references.
        """
        total_size = 0
        total_size += sys.getsizeof(self.black)
        total_size += sys.getsizeof(self.white)
        total_size += sys.getsizeof(self.no_move)
        total_size += sys.getsizeof(self.size)
        total_size += sys.getsizeof(self.history_length)
        total_size += sys.getsizeof(self.move_history)
        total_size += sys.getsizeof(self.pass_idx)
        total_size += sys.getsizeof(self.full_history)
        total_size += sys.getsizeof(self.max_moves)
        total_size += sys.getsizeof(self.full_black_stones_history)
        total_size += sys.getsizeof(self.full_black_liberty_history)
        total_size += sys.getsizeof(self.full_white_stones_history)
        total_size += sys.getsizeof(self.full_white_liberty_history)
        total_size += sys.getsizeof(self.previous_board)
        total_size += sys.getsizeof(self.current_board)
        total_size += sys.getsizeof(self.komi)
        total_size += sys.getsizeof(self.black_string_count)
        total_size += sys.getsizeof(self.white_string_count)
        total_size += sys.getsizeof(self.strings)
        total_size += sys.getsizeof(self.string_board)
        total_size += sys.getsizeof(self.current_player)
        total_size += sys.getsizeof(self.black_to_go_next)
        total_size += sys.getsizeof(self.white_to_go_next)
        total_size += sys.getsizeof(self.black_liberties)
        total_size += sys.getsizeof(self.white_liberties)
        total_size += sys.getsizeof(self.available_liberties)
        total_size += sys.getsizeof(self.legal_black_moves)
        total_size += sys.getsizeof(self.legal_white_moves)
        total_size += sys.getsizeof(self.illegal_pos_by_ko)
        return total_size

    def get_state_w_libs_old(self):
        """ Return the state of the game in a nparray that can be processed by the wingobot NN. """
        if self.current_player == self.black:
            additional_layers = [self.black_to_go_next, self.black_liberties, self.white_liberties]
        else:
            additional_layers = [self.white_to_go_next, self.white_liberties, self.black_liberties]
        states_w_libs = [layer for layer in self.full_history]
        for layer in additional_layers:
            states_w_libs.append(layer)
        #state_w_libs = np.append(self.full_history, additional_layers)
        return np.asarray(states_w_libs)

    def get_state_w_libs(self):
        """ Return the state of the game in a nparray that can be processed by the wingobot NN. """
        # Store the state in layers, as F1, E1, ...., F8, E8, I-friendly, L-friendly, L-enemy
        layers = []
        curr_move_number = len(self.move_history)
        # If there have been fewer moves than the requested history length, pad with zero filled layers.
        num_pads = abs(min(0, curr_move_number - self.history_length))
        for _ in range(num_pads):
            layers.append(np.zeros((self.size, self.size), dtype=np.int8))
            layers.append(np.zeros((self.size, self.size), dtype=np.int8))
        # Identify the range of moves to append to the state list. Then add the stone state layers.
        range_low = max(0, curr_move_number - self.history_length)
        range_high = curr_move_number
        for idx in range(range_low, range_high):
            if self.current_player == self.black:
                layers.append(self.full_black_stones_history[idx])
                layers.append(self.full_white_stones_history[idx])
            else:
                layers.append(self.full_white_stones_history[idx])
                layers.append(self.full_black_stones_history[idx])
        # Add the identity layer.
        if self.current_player == self.black:
            layers.append(self.black_to_go_next)
        else:
            layers.append(self.white_to_go_next)
        # Add the liberty layers.
        if self.current_player == self.black:
            layers.append(self.full_black_liberty_history[curr_move_number])
            layers.append(self.full_white_liberty_history[curr_move_number])
        else:
            layers.append(self.full_white_liberty_history[curr_move_number])
            layers.append(self.full_black_liberty_history[curr_move_number])
        # Convert the layers to a numpy array and return that.
        return np.asarray(layers)

    def get_legal_moves(self):
        """ Return a numpy array of the legal moves, reshaped to (size**2 + 1). """
        if self.current_player == self.black:
            return np.append(np.reshape(self.legal_black_moves, (self.size**2)), [1])
        else:
            return np.append(np.reshape(self.legal_white_moves, (self.size**2)), [1])

    def ogs_score(self):
        """
        Use the OGS scoring algorithm, from the ogs-estimator repo.
        """
        flat_int_curr_board = np.reshape(self.current_board, self.size**2).astype(int).tolist()
        trials = 1000
        tolerance = 0.4
        final_score = ogs_estimate(self.size, self.size, flat_int_curr_board,
                                   self.current_player, trials, tolerance) - self.komi
        # Return a string in SGF format.
        if final_score > 0:
            return "B+" + str(final_score)
        else:
            return "W+" + str(final_score*-1)

    def fast_score(self):
        """
        Just record if white or black one.
        :return: the score from the perspective of the white or black player (either B+1000 or W+1000).
        """
        final_score = np.sum(self.current_board) - self.komi  # This assumes white = -1, black = 1, and spaces = 0.
        if final_score < 0:
            return "W+1000"
        else:
            return "B+1000"

    def tromp_taylor_score(self):
        """
        Scores a game using Tromp Taylor rules: https://codegolf.stackexchange.com/questions/6693/score-a-game-of-go
        Basically, process every empty space as a string.
        If this string is only surrounded by one player (and the edge of the board), it is that player's territory.
        :return: the total score (komi adjusted) from the perspective of the winning player (e.g. B+2.5, or W+4.5).
        """
        # Make a copy of the board and reshape it.
        board = np.reshape(self.current_board.copy(), self.size**2)
        # Keep a list of visited indices.
        visited = []
        # Identify all the strings of empty spaces on the board.
        for idx in range(self.size**2):
            # Ignore intersections we have already gone over.
            if idx in visited:
                continue
            # If no stone is there, identify the full 'string' of empty spaces.
            if board[idx] == 0:
                row = idx // self.size
                col = idx % self.size
                empty_string = self.get_string_at_idx_on_board(row, col)
                # Get a list of stones of each type that are adjacent to the empty string.
                black_adjacent_strings = self.get_opponent_strings_of_value(empty_string, self.black)
                white_adjacent_strings = self.get_opponent_strings_of_value(empty_string, self.white)
                # If only plus one strings are adjacent, set the space to all plus ones.
                if black_adjacent_strings and not white_adjacent_strings:
                    for empty_idx in empty_string.indices:
                        board[empty_idx] = self.black
                # If only minus one strings are adjacent, set the space to all minus ones.
                if white_adjacent_strings and not black_adjacent_strings:
                    for empty_idx in empty_string.indices:
                        board[empty_idx] = self.white
                # Add the indices in the empty string to the visited list.
                visited.extend(empty_string.indices)
        # Total score is represented by sum of each players stones and their territory (now labeled with their value).
        black_total = 0
        white_total = self.komi
        for stone in board:
            if stone == self.black:
                black_total += 1
            elif stone == self.white:
                white_total += 1
        # Return a string in SGF format.
        if black_total > white_total:
            diff = black_total - white_total
            return "B+" + str(diff)
        else:
            diff = white_total - black_total
            return "W+" + str(diff)

    def get_opponent_strings_of_value(self, a_string, opponent_value):
        """
        Get all of the strings adjacent to a_string with the given value.
        :param a_string: a dict holding {"indices": [list of string indices], "liberties": [list of string liberties]}
        :param opponent_value: the value of the opponent stone of interest (could also be an empty space).
        :return: a list of strings of the given value that are adjacent to a_string.
        """
        # First, loop through the indices of self string to get a list of all touching opponent stone indices.
        visited = []
        opponents = []
        for string_idx in a_string.indices:  # Get value of current stone.
            # Get values for the top bottom left and right intersections.
            adjacent_intersections = self.get_adjacent_intersections(string_idx[0], string_idx[1])
            # Add connected stones to the list if they have not been visited.
            for adjacent_idx, adjacent_value in adjacent_intersections:
                if adjacent_value == opponent_value and adjacent_idx not in visited:
                    opponent_string = self.get_string_at_idx_on_board(adjacent_idx[0], adjacent_idx[1])
                    visited.extend(opponent_string.indices)
                    opponents.append(opponent_string)
        return opponents

    def get_string_at_idx_on_board(self, row, col):
        """
        Given an index and a board, return a string (list of indices) and its liberty
        count based on the connected stones (or spaces) of the same value.
        :param row: the row index.
        :param col: the column index.
        :return: a dict holding {"indices": [], "liberty_count": int}
        """
        # Loop through all stones in the current string.
        current_string = [[row, col]]
        liberties = []
        for s_row, s_col in current_string:
            # Get values for the top bottom left and right intersections.
            center = self.current_board[s_row, s_col]
            adjacent_intersections = self.get_adjacent_intersections(s_row, s_col)
            # Loop through the four adjacent intersections.
            for adjacent_idx, adjacent_value in adjacent_intersections:
                # Add friendly connected stones to the list if they have not been visited.
                if adjacent_value == center and adjacent_idx not in current_string:
                    current_string.append(adjacent_idx)
                # Increment the count of liberties, for any uncounted liberties.
                if adjacent_value == 0 and adjacent_idx not in liberties:
                    liberties.append(adjacent_idx)
        return String(string_id="TBD", indices=current_string, liberties=liberties)

    def get_adjacent_intersections(self, row, col):
        """
        The the values at the four intersections adjacent to this position. Do not include values off the board.
        :param row: the row index.
        :param col: the column index.
        :return:
        """
        # Build a list of the adjacent intersections and their values.
        adjacent_intersections = []
        # Get the value of the intersection above [row, col].
        top_row = row - 1
        top_col = col
        if top_row >= 0:
            top_value = self.current_board[top_row, top_col]
            adjacent_intersections.append(([top_row, top_col], top_value))
        # Get the value of the intersection below [row, col].
        bottom_row = row + 1
        bottom_col = col
        if bottom_row < self.size:
            bottom_value = self.current_board[bottom_row, bottom_col]
            adjacent_intersections.append(([bottom_row, bottom_col], bottom_value))
        # Get the value of the intersection to the left of [row, col].
        left_row = row
        left_col = col - 1
        if left_col >= 0:
            left_value = self.current_board[left_row, left_col]
            adjacent_intersections.append(([left_row, left_col], left_value))
        # Get the value of the intersection to the right of [row, col].
        right_row = row
        right_col = col + 1
        if right_col < self.size:
            right_value = self.current_board[right_row, right_col]
            adjacent_intersections.append(([right_row, right_col], right_value))
        # Return the intersections.
        return adjacent_intersections

    def copy(self):
        """
        Create a duplicate of this Goban and its current state / history.
        This duplicate may be used to explore new board positions during a MCTS, for example.
        """
        return copy.deepcopy(self)

    def __move_is_legal(self, row, col):
        """
        Verify that the move at the given row and column is legal for the current player to make.
        :param row: the row index of the move.
        :param col: the column index of the move.
        :return: True if the move is legal, false otherwise.
        """
        if self.current_player == self.black:
            if self.legal_black_moves[row][col] != 1:
                print("WARNING: Black attempted an illegal move at row: ", row, " col: ", col)
                return False
            else:
                return True
        elif self.current_player == self.white:
            if self.legal_white_moves[row][col] != 1:
                print("WARNING: White attempted an illegal move at row: ", row, " col: ", col)
                return False
            else:
                return True

    def __increment_liberties_adjacent_to(self, row, col):
        """
        Increment the count of new liberties that would be awarded to a stone played in the adjacent spots.
        Call this when a stone has been removed from an index.
        If the count moves from 0 to 1 check the liberty count of the adjacent strings to determine move legality.
        If the count moves up from 1 then playing in this spot will be legal for both players.
        """
        adjacent_spots = self.get_adjacent_intersections(row, col)
        for spot_idx, _ in adjacent_spots:
            spot_row, spot_col = spot_idx
            self.available_liberties[spot_row, spot_col] += 1
            # If there is no stone here, and moving in this spot would provide a liberty, its a legal move.
            if self.current_board[spot_row, spot_col] == 0:
                self.__set_legal_for_black(spot_row, spot_col)
                self.__set_legal_for_white(spot_row, spot_col)

    def __decrement_liberties_adjacent_to(self, row, col):
        """
        Decrement the count of new liberties that are available to a stone played in the adjacent spots.
        If the count moves to 0, a move there will only be legal if it results in a capture.
        For any spots that would award one or more liberties w/out capture, moving there is always legal.
        """
        adjacent_spots = self.get_adjacent_intersections(row, col)
        for spot_idx, _ in adjacent_spots:
            spot_row, spot_col = spot_idx
            self.available_liberties[spot_row, spot_col] -= 1
            if self.available_liberties[spot_row, spot_col] == 0 and self.current_board[spot_row, spot_col] == 0:
                self.__set_legality_dependent_on_liberties(spot_row, spot_col)

    def __set_legality_dependent_on_liberties(self, row, col):
        """
        Set the legality of a move in this position for the black and white players based on
        their liberty counts in the adjacent spots. If a move here would result in the capture of an opponent,
        or if the move would connect to a friendly string with more than one liberty, it is legal.
        """
        adjacent_spots = self.get_adjacent_intersections(row, col)
        results_in_black_liberties = False
        results_in_white_liberties = False
        for spot_idx, spot_value in adjacent_spots:
            if spot_value == 0:
                results_in_black_liberties = True
                results_in_white_liberties = True
                continue
            spot_row, spot_col = spot_idx
            string_id = self.string_board[spot_row, spot_col]
            lib_count = self.strings[string_id].liberty_count()
            if spot_value == self.black:
                if lib_count > 1:
                    results_in_black_liberties = True
                elif lib_count == 1:
                    results_in_white_liberties = True
            elif spot_value == self.white:
                if lib_count > 1:
                    results_in_white_liberties = True
                elif lib_count == 1:
                    results_in_black_liberties = True
        # If the move results in white or black liberties (either through capture or connection, its legal).
        if results_in_black_liberties:
            self.__set_legal_for_black(row, col)
        else:
            self.__set_illegal_for_black(row, col)
        if results_in_white_liberties:
            self.__set_legal_for_white(row, col)
        else:
            self.__set_illegal_for_white(row, col)

    def __set_illegal_for_black(self, row, col):
        """
        Set the legality of a move at this position to 0 (ie illegal) for the black player.
        :param row: the row index of the move to make illegal.
        :param col: the column index of the move to make illegal.
        """
        self.legal_black_moves[row, col] = 0

    def __set_illegal_for_white(self, row, col):
        """
        Set the legality of a move at this position to 0 (ie illegal) for the white player.
        :param row: the row index of the move to make illegal.
        :param col: the column index of the move to make illegal.
        """
        self.legal_white_moves[row, col] = 0

    def __set_legal_for_black(self, row, col):
        """
        Set the legality of a move at this position to 1 (ie legal) for the black player.
        :param row: the row index of the move to make legal.
        :param col: the column index of the move to make legal.
        """
        self.legal_black_moves[row, col] = 1

    def __set_legal_for_white(self, row, col):
        """
        Set the legality of a move at this position to 1 (ie legal) for the white player.
        :param row: the row index of the move to make legal.
        :param col: the column index of the move to make legal.
        """
        self.legal_white_moves[row, col] = 1

    def __place_stone_at_index(self, row, col):
        """
        This function places a stone for the current player at the given index.
        It adjusts the strings and liberty counts as necessary, but does NOT remove stones.
        It also adjusts the legality of moves.
        :param row: the row index of the stone to place.
        :param col: the column index of the stone to place.
        :return: a list of the opponent string ids that have been captured.
        """
        # Place the stone at this index, and then get the string that is at this index. This includes a liberty count.
        self.current_board[row, col] = self.current_player
        new_string = self.get_string_at_idx_on_board(row, col)
        # Moving here will now be illegal for both players.
        self.__set_illegal_for_black(row, col)
        self.__set_illegal_for_white(row, col)
        # Update the liberty count and string board for this string.
        if self.current_player == self.black:
            self.black_string_count += 1
            new_string_id = "B" + str(self.black_string_count)
            new_string.string_id = new_string_id
            for s_row, s_col in new_string.indices:
                self.string_board[s_row, s_col] = new_string_id
                self.black_liberties[s_row, s_col] = new_string.liberty_count()
            self.strings[new_string_id] = new_string
            if new_string.liberty_count() == 1:
                lib_row, lib_col = new_string.liberties[0]
                self.__set_legality_dependent_on_liberties(lib_row, lib_col)
        elif self.current_player == self.white:
            self.white_string_count += 1
            new_string_id = "W" + str(self.white_string_count)
            for s_row, s_col in new_string.indices:
                self.string_board[s_row, s_col] = new_string_id
                self.white_liberties[s_row, s_col] = new_string.liberty_count()
            self.strings[new_string_id] = new_string
            if new_string.liberty_count() == 1:
                lib_row, lib_col = new_string.liberties[0]
                self.__set_legality_dependent_on_liberties(lib_row, lib_col)
        # Identify the adjacent enemy strings, to update their liberty counts (decrement by one).
        opponent_value = self.white if self.current_player == self.black else self.black
        opponent_string_ids = []
        for adj_idx, adj_value in self.get_adjacent_intersections(row, col):
            if adj_value == opponent_value:
                adj_row, adj_col = adj_idx
                string_id = self.string_board[adj_row, adj_col]
                if string_id != "EMPTY" and string_id not in opponent_string_ids:
                    opponent_string_ids.append(string_id)
        captured_opponents = []
        for opp_string_id in opponent_string_ids:
            opp_string = self.strings[opp_string_id]
            opp_string.remove_liberty(row, col)
            new_opp_lib_count = opp_string.liberty_count()
            for s_row, s_col in opp_string.indices:
                if opponent_value == self.black:
                    self.black_liberties[s_row, s_col] = new_opp_lib_count
                elif opponent_value == self.white:
                    self.white_liberties[s_row, s_col] = new_opp_lib_count
            if new_opp_lib_count == 0:
                captured_opponents.append(opp_string_id)
            elif new_opp_lib_count == 1:
                lib_row, lib_col = opp_string.liberties[0]
                self.__set_legality_dependent_on_liberties(lib_row, lib_col)
        self.__decrement_liberties_adjacent_to(row, col)
        return captured_opponents

    def __remove_stone_at_index(self, row, col):
        """
        This function removes a stone from current board at the given index.
        It will increment the liberty count of adjacent enemy strings.
        :param row:
        :param col:
        :return:
        """
        # Get the value of the stone at this index, to update its adjacent opponents liberty counts (increment by one).
        stone_value = self.current_board[row, col]
        # Clear that position.
        self.current_board[row, col] = 0
        self.string_board[row, col] = "EMPTY"
        # Get the opponent strings.
        opponent_value = self.white if stone_value == self.black else self.black
        opponent_string_ids = []
        for adj_idx, adj_value in self.get_adjacent_intersections(row, col):
            if adj_value == opponent_value:
                adj_row, adj_col = adj_idx
                string_id = self.string_board[adj_row, adj_col]
                if string_id not in opponent_string_ids:
                    opponent_string_ids.append(string_id)
        for opp_string_id in opponent_string_ids:
            opp_string = self.strings[opp_string_id]
            opp_string.add_liberty(row, col)
            new_lib_count = opp_string.liberty_count()
            for s_row, s_col in opp_string.indices:
                if opponent_value == self.black:
                    self.black_liberties[s_row, s_col] = new_lib_count
                elif opponent_value == self.white:
                    self.white_liberties[s_row, s_col] = new_lib_count
        # Increment the available liberties for a stone played at this spot by one.
        self.__increment_liberties_adjacent_to(row, col)

    def __remove_string(self, string_id):
        """
        Remove a string, one stone at a time.
        :param string_id: the id of the string to remove.
        :return the indices of the stone removed.
        """
        # Remove each stone in this string. Then delete it from the records.
        stones_removed = []
        for s_row, s_col in self.strings[string_id].indices:
            self.__remove_stone_at_index(s_row, s_col)
            stones_removed.append([s_row, s_col])
        del self.strings[string_id]
        # Set the legality of moving at the freed indices to legal if more than one stone is freed...
        if len(stones_removed) > 1:
            for s_row, s_col in stones_removed:
                self.__set_legal_for_black(s_row, s_col)
                self.__set_legal_for_white(s_row, s_col)
        # Otherwise, if the number of stones removed is only equal to one, the legality is dependent on liberty counts.
        elif len(stones_removed) == 1:
            self.__set_legality_dependent_on_liberties(stones_removed[0][0], stones_removed[0][1])
        return stones_removed

    def __reset_ko(self):
        """
        Reset the position that was an illegal move based on the Ko rule, if and only if the move will
        actually be legal based on liberty counts and capture potential.
        :return:
        """
        if self.illegal_pos_by_ko:
            ko_row, ko_col = self.illegal_pos_by_ko
            self.__set_legality_dependent_on_liberties(ko_row, ko_col)
            self.illegal_pos_by_ko = None

    def __set_ko(self, row, col):
        """
        Set the position that will be illegal for the NEXT player based on the Ko rule.
        :param row: the row index of the move.
        :param col: the col index of the move.
        """
        self.illegal_pos_by_ko = [row, col]
        if self.current_player == self.black:
            self.__set_illegal_for_white(row, col)
        elif self.current_player == self.white:
            self.__set_illegal_for_black(row, col)

    def __make_pass(self):
        self.__reset_ko()
        self.previous_board = self.current_board.copy()
        self.__update_total_histories(row=self.size, col=self.size)
        self.set_next_player()

    def make_move(self, row, col):
        """
        The current player will attempt to make a move at the given row and column.
        If the move is illegal, a pass is made and the board state is not changed but a warning is printed.
        :param row: the row index of the move.
        :param col: the column index of the move.
        :return: True if the move was successful / valid, False otherwise.
        """
        # Check if this move is a pass.
        if row == 13 or col == 13:
            self.__make_pass()
            #print("Making a pass. Current player: ", self.current_player)
            return False

        # First, check if the move is in a legal spot.
        if not self.__move_is_legal(row, col):
            print("Attempting an illegal move at row: ", row, " col: ", col, ".")
            self.__make_pass()
            return False

        # Reset the Ko.
        self.__reset_ko()

        # Copy the current single board into the previous single board.
        self.previous_board = self.current_board.copy()

        # Add the move to the current board.
        captured_opponents = self.__place_stone_at_index(row, col)

        # Remove the captured opponents. This will add liberties to strings with newly freed space.
        removed_stones = []
        for captured_string_id in captured_opponents:
            removed_stones.extend(self.__remove_string(captured_string_id))

        # If there was only one stone removed, then this move will be illegal the next turn based on the Ko rule.
        if len(removed_stones) == 1:
            ko_row, ko_col = removed_stones[0]
            self.__set_ko(ko_row, ko_col)

        # Update the stored histories, including the list of moves, black and white board states, and liberties.
        self.__update_total_histories(row, col)

        # Make it the next players turn.
        self.set_next_player()
        return True

    def __update_total_histories(self, row, col):
        """
        Store the current mono board states (ie location of black and white stones),
        as well as the black and white history counts, at the index of the current move.
        """
        history_idx = len(self.move_history)
        self.full_black_liberty_history[history_idx] = self.black_liberties
        self.full_white_liberty_history[history_idx] = self.white_liberties
        self.full_black_stones_history[history_idx] = self.__get_mono_stone_board(self.black)
        self.full_white_stones_history[history_idx] = self.__get_mono_stone_board(self.white)
        self.move_history.append([row, col])

    def __update_full_history(self):
        """
        This function shifts the full_history array by two,
        pushing out the oldest move and adding the newest to the stack.
        It uses the newly updated current board state to accomplish this.
        """
        updated_history = []
        for idx in range(2, self.history_length*2, 2):
            updated_history.append(self.full_history[idx + 1])
            updated_history.append(self.full_history[idx])
        single_black_board = self.__get_mono_stone_board(self.black)
        single_white_board = self.__get_mono_stone_board(self.white)
        if self.current_player == self.black:
            updated_history.append(single_white_board)
            updated_history.append(single_black_board)
        else:
            updated_history.append(single_black_board)
            updated_history.append(single_white_board)
        self.full_history = np.asarray(updated_history)

    def __get_mono_stone_board(self, stone_color):
        mono_board = np.zeros((self.size, self.size))
        for row in range(self.size):
            for col in range(self.size):
                if self.current_board[row, col] == stone_color:
                    mono_board[row, col] = stone_color
        return mono_board

    def set_next_player(self):
        if self.current_player == self.black:
            self.current_player = self.white
        else:
            self.current_player = self.black

    def get_current_board(self):
        """
        """
        return self.current_board


    def get_current_str_ids(self):
        """
        """
        return self.string_board

    def get_current_lib_counts(self):
        """
        """
        return self.black_liberties + self.white_liberties, np.reshape(self.available_liberties, (self.size, self.size))

    def save_game_to_sgf(self, outfile):
        """
        Given a list of moves ([B, W, B, W...]) and the winner of a game, save this to the provided output file using SGF.
        :param outfile:
        :return:
        """
        # Get the score.
        game_result = self.tromp_taylor_score()
        #game_result = self.fast_score()

        moves_list = self.move_history

        # Get the index given the alphabet values of the move.
        move_code_map = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g",
                         7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m"}
        # Add the header.
        lines = ["(;GM[1]FF[4]CA[UTF-8]", "RU[Chinese]SZ[13]KM[7.5]TM[600]",
                 "PW[wingobot]PB[wingobot]WR[00]BR[00]DT[2020-07-30]PC[wingo-desktop]" + "RE[" + game_result + "]GN[007]"]
        # For every move, starting with black, add it to the list of lines.
        player = "B"  # Alternates between B and W.
        for move in moves_list:
            if move == self.pass_idx:
                move_code = ""  # For a pass.
            else:
                row, col = move
                move_code = move_code_map[row] + move_code_map[col]
            move_str = ";" + player + "[" + move_code + "]"
            lines.append(move_str)
            if player == "B":
                player = "W"
            else:
                player = "B"
        # Add a closing paren.
        lines.append(")")
        # Write every line to the file.
        with open(outfile, "w") as f:
            for line in lines:
                f.write(line)

    def load_game_from_sgf(self, sgf_file):
        """
        Initialize a game state by reading moves from an SGF file.
        :param sgf_file:
        :return:
        """
        reverse_move_code_map = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g",
                                 7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m"}
        move_code_map = {value: key for key, value in reverse_move_code_map.items()}
        with open(sgf_file, "r") as sgf_f:
            content = sgf_f.readlines()[0].split(";")
        move_list = []
        for item in content:
            if item.startswith("B[") and item != "B[]":
                row = move_code_map[item[2]]
                col = move_code_map[item[3]]
                move_list.append((row, col))
            elif item == "B[]":
                row = 13
                col = 13
                move_list.append((row, col))
            elif item.startswith("W[") and item != "W[]":
                row = move_code_map[item[2]]
                col = move_code_map[item[3]]
                move_list.append((row, col))
            elif item == "W[]":
                row = 13
                col = 13
                move_list.append((row, col))
        for row, col in move_list:
            self.make_move(row, col)

    def load_game_from_moves_list(self, moves_list):
        """
        :param moves_list: a list of shape (N, 2) holding the row and col of each move.
        """
        for move in moves_list:
            row, col = move
            self.make_move(row, col)

    def print_board(self):
        """
        Print a board to stdout, using emojis to represent the board and stones.
        :param black_board:
        :param white_board:
        :return:
        """
        header = "  "
        for x in range(self.size):
            header += str(x) + " "
        print(header)
        for row in range(self.size):
            current_line = str(row) + " "
            for col in range(self.size):
                if self.current_board[row, col] == 1:
                    #current_line += " \U000026AB "
                    current_line += " \U0001F535 "
                elif self.current_board[row, col] == -1:
                    #current_line += " \U000026AA "
                    current_line += " \U0001F534 "
                else:
                    current_line += " \U00002795 "
            print(current_line)
        print("\n\n")


def print_board(black_board, white_board):
    # TODO Definitely keep this as a class method, but just use the single board. It will be simpler that way.
    """
    Print a board to stdout, using emojis to represent the board and stones.
    :param black_board:
    :param white_board:
    :return:
    """
    current_line = ""
    for idx in range(169):
        if black_board[idx] == 1:
            #current_line += " \U000026AB "
            current_line += "\U0001F535"
        elif white_board[idx] == 1:
            #current_line += "\U000026AA "
            current_line += "\U0001F534"
        else:
            #current_line += " \U00002795 "
            current_line += "+"

        if (idx+1) % 13 == 0:
            print(current_line)
            current_line = ""

    print("\n\n")


def transform_board_and_policy(board, policy):
    # TODO I think this should become a Goban class method but right now its used by the SL training scripts.
    #  The SL scripts should probably be reworked to take advantage of the Goban class.
    #  This will require writing a new initializer that loads the game state from an HDF5 file.
    """
    Randomly rotate and flip a board.
    :param board: a Nx13x13 numpy array
    :param policy: a 170 element list (1 for selected move, 0 for the rest)
    :return:
    """
    num_rotations = random.choice([0, 1, 2, 3])
    flip = random.choice([True, False])
    new_board = np.rot90(board, num_rotations, axes=(1, 2))
    new_policy = np.asarray(policy[:-1]).reshape((13, 13))
    new_policy = np.rot90(new_policy, num_rotations)
    if flip:
        new_board = np.flip(new_board, axis=2)
        new_policy = np.flip(new_policy, axis=1)
    if policy[-1] == 1:
        new_policy = policy
    else:
        new_policy = np.reshape(new_policy, 169).tolist()
        new_policy.append(0)
    return new_board, new_policy
