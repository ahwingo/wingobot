import numpy as np
import random

def tromp_taylor_score(board):
    """
    Scores a game using Tromp Taylor rules: https://codegolf.stackexchange.com/questions/6693/score-a-game-of-go
    Basically, process every empty space as a string.
    If this string is only surrounded by one player (and the edge of the board), it is that player's territory.
    :param board: a 13x13 matrix representing board state, consiting of -1, 0, or 1 values.
    :return: the total score (not including komi), from the perspective of the player whose stones are given as '1'.
    """
    # Make a copy of the board and reshape it.
    board = np.reshape(board.copy(), 169)

    # Keep a list of visited indices.
    visited = []

    # Identify all the strings of empty spaces on the board.
    for idx in range(169):
        # Ignore intersections we have already gone over.
        if idx in visited:
            continue

        # If no stone is there, identify the full 'string' of empty spaces.
        if board[idx] == 0:
            empty_string = get_string_at_idx_on_board(idx, board)

            # Get a list of stones of each type that are adjacent to the empty string.
            plus_one_strings = get_opponent_strings_of_value(empty_string, board, 1)
            minus_one_strings = get_opponent_strings_of_value(empty_string, board, -1)

            # If only plus one strings are adjacent, set the space to all plus ones.
            if plus_one_strings and not minus_one_strings:
                for empty_idx in empty_string["indices"]:
                    board[empty_idx] = 1
            # If only minus one strings are adjacent, set the space to all minus ones.
            if minus_one_strings and not plus_one_strings:
                for empty_idx in empty_string["indices"]:
                    board[empty_idx] = -1

            # Add the indices in the empty string to the visited list.
            visited.extend(empty_string)

    # The total score is represented by the sum of board intersection values.
    total_score = 0
    for x in board:
        total_score += x
    return total_score


def get_single_scorable_board_from_state(board_state_a, board_state_b):
    board = [board_state_a[idx] - board_state_b[idx] for idx in range(169)]
    return board


def get_single_storable_board_from_state(board_state_black, board_state_white):
    """
    I have not looked at this in a while, but it looks like black == 1 and white == 2...
    :param board_state_black:
    :param board_state_white:
    :return:
    """
    board = [1*board_state_black[idx] + 2*board_state_white[idx] for idx in range(169)]
    return board


def get_full_state_from_byte_board_history(byte_boards, player_to_go_next):
    """
    Returns a 13x13x17 board state holding the last handful of moves for both players, plus a tensor for who goes next.
    :param byte_boards: the last 8 board positions
    :param player_to_go_next: "black" or "white"
    """
    if player_to_go_next is "black":
        curr_player_value = 1
        next_player_value = 2
    else:
        curr_player_value = 2
        next_player_value = 1

    player_to_go_value = 1 if player_to_go_next is "black" else 0

    boards = []
    for byte_board in byte_boards:
        curr_player_board = [1 if int(x) is curr_player_value else 0 for x in byte_board]
        next_player_board = [1 if int(x) is next_player_value else 0 for x in byte_board]
        boards.append(curr_player_board)
        boards.append(next_player_board)
    player_to_go_board = [player_to_go_value]*169
    boards.append(player_to_go_board)

    return boards


def get_string_at_idx_on_board(idx, board):
    # Loop through all stones in the current string.
    current_string = [idx]
    liberties = []
    for string_idx in current_string:

        # Get values for the top bottom left and right intersections.
        center = board[string_idx]
        adjacent_intersections = get_adjacent_intersections(string_idx, board)

        # Loop through the four adjacent intersections.
        for adjacent_idx, adjacent_value in adjacent_intersections:

            # Add friendly connected stones to the list if they have not been visited.
            if adjacent_value == center and adjacent_idx not in current_string:
                current_string.append(adjacent_idx)

            # Increment the count of liberties, for any uncounted liberties.
            if adjacent_value == 0 and adjacent_idx not in liberties:
                liberties.append(adjacent_idx)

    return {"indices": current_string, "liberty_count": len(liberties)}


def get_string_liberties(a_string, board):
    liberties = []
    for stone in a_string:
        # Get values for the top bottom left and right intersections.
        adjacent_intersections = get_adjacent_intersections(stone, board)
        for adjacent_idx, adjacent_value in adjacent_intersections:
            # Increment the count of liberties, for any uncounted liberties.
            if adjacent_value == 0 and adjacent_idx not in liberties:
                liberties.append(adjacent_idx)
    return len(liberties)


def get_adjacent_intersections(center_idx, board):

    row = center_idx // 13
    col = center_idx % 13

    # Get values for the top bottom left and right intersections.
    adjacent_intersections = []
    top_idx = int(13 * (row - 1) + col)
    top = board[top_idx] if row > 0 else -2
    adjacent_intersections.append((top_idx, top))
    bottom_idx = int(13 * (row + 1) + col)
    bottom = board[bottom_idx] if row < 12 else -2
    adjacent_intersections.append((bottom_idx, bottom))
    left_idx = int(13 * row + col - 1)
    left = board[left_idx] if col > 0 else -2
    adjacent_intersections.append((left_idx, left))
    right_idx = int(13 * row + col + 1)
    right = board[right_idx] if col < 12 else -2
    adjacent_intersections.append((right_idx, right))

    # Return the intersections.
    return adjacent_intersections


def get_opponent_strings_of_value(a_string, board, opponent_value):
    # First, loop through the indices of self string to get a list of all touching opponent stone indices.
    visited = []
    opponents = []
    for string_idx in a_string["indices"]:            # Get value of current stone.

        # Get values for the top bottom left and right intersections.
        center = board[string_idx]
        adjacent_intersections = get_adjacent_intersections(string_idx, board)

        # Add connected stones to the list if they have not been visited.
        for adjacent_idx, adjacent_value in adjacent_intersections:
            if adjacent_value == opponent_value and adjacent_idx not in visited:
                opponent_string = get_string_at_idx_on_board(adjacent_idx, board)
                visited.extend(opponent_string["indices"])
                opponents.append(opponent_string)

    return opponents


def board_pos_after_move(current_board, previous_board, move_idx, stone_value):
    """
    Identifies what the new board will look like after a move is played at the given index, if its legal.
    :param current_board: a 13 x 13 array holding values of -1, 0, or 1.
    :param previous_board: a 13 x 13 array holding values of -1, 0, or 1.
    :param move_idx: an integer in the range of 0 to 168 (this function does not handle passes).
    :param stone_value: the stone value of the playing player (either 1 or -1).
    :return: a dictionary showing if the move is legal (boolean) and the resulting board state (13 x 13 array).
    """
    # If the move is out of bounds, return false.
    if (move_idx < 0) or (move_idx >= 169):
        return {"move_legal": False, "board_outcome": current_board}

    # If the move is on top of a stone that is already on the board, it is illegal!!!
    if current_board[move_idx] != 0:
        return {"move_legal": False, "board_outcome": current_board}

    # The proposed board position is the current position plus the new move.
    proposed_board = current_board.copy()
    proposed_board[move_idx] = stone_value

    # Get indices of new string made by last move. {"indices": [indices], "liberty_count" int}
    new_move_string = get_string_at_idx_on_board(move_idx, proposed_board)

    # Get list of all opponent strings that touch the new move string. [{"string": [indices], "liberty_count" int}]
    opponent_strings = get_opponent_strings_of_value(new_move_string, proposed_board, -1*stone_value)

    # Remove captured opponents.
    for opp_string in opponent_strings:
        if opp_string["liberty_count"] == 0:
            for opp_string_idx in opp_string["indices"]:
                proposed_board[opp_string_idx] = 0

    # Now that recently captured opponents are removed, check the liberty count of the newly played move / string.
    new_move_string = get_string_at_idx_on_board(move_idx, proposed_board)

    # If the new move resulted in a self capture, it is illegal.
    if new_move_string["liberty_count"] == 0:
        return {"move_legal": False, "board_outcome": current_board}

    # The move may have broken the Ko rule. Check if the new proposed board state matches the previous board state.
    all_intersections_equal = True
    for idx in range(169):
        if previous_board[idx] != proposed_board[idx]:
            all_intersections_equal = False
            break
    # If the boards match completely, the Ko rule was broken and the move is illegal.
    if all_intersections_equal:
        return {"move_legal": False, "board_outcome": current_board}

    # Otherwise, the move was completely legal! Return the new board state, with captured stones removed.
    return {"move_legal": True, "board_outcome": proposed_board}


def get_liberty_counts_from_board(current_board, friendly_value):
    """
    Returns two boards, one for friendly stone liberty counts, one for non friendly liberty counts.
    Counts are returned as string lib count / 8.0, where string lib count is capped at 8.
    This is done to keep these layers at the same scale as the other binary inputs.
    """
    current_board = np.reshape(current_board, 169).tolist()
    string_board = [-1]*169  # If a string is here, it will store the id of this string.
    string_count = 0
    string_liberty_counts = {}  # Key: string id, value: liberty count.A
    liberties_board = [[] for _ in range(169)]
    friendly_lib_count = [0]*169
    enemy_lib_count = [0]*169
    enemy_value = 2 if friendly_value == 1 else 1  # The current board is made of 0s (empty) 1s (black) & 2s (white).

    for idx in range(169):
        # If this stone has already been considered, continue.
        if string_board[idx] != -1:
            continue
        # If a stone is here, it is a string.
        if current_board[idx] != 0:
            stone_value = current_board[idx]
            string_board[idx] = string_count
            string_liberty_counts[string_count] = 0  # Since this is the first time we have encountered this string, set its lib count to zero.
            current_string = [idx]  # This string will contain all indices of connected stones.
            for stone_idx in current_string:
                adjacent_stones = get_adjacent_intersections(stone_idx, current_board)
                for adjacent_stone in adjacent_stones:
                    adj_stone_idx = adjacent_stone[0]
                    adj_stone_value = adjacent_stone[1]
                    # If the adjacent stone value matches the strings value,
                    # and the stone has not been considered yet, add it to the string.
                    if adj_stone_value == stone_value and string_board[adj_stone_idx] == -1:
                        current_string.append(adj_stone_idx)
                        string_board[adj_stone_idx] = string_count
                    # Otherwise, if the adjacent stones value is 0, it is a liberty.
                    if adj_stone_value == 0 and string_count not in liberties_board[adj_stone_idx]:
                        string_liberty_counts[string_count] += 1
                        liberties_board[adj_stone_idx].append(string_count)
            # Set the value of the liberties for this string.
            lib_value = min(8, string_liberty_counts[string_count]) / 8.0
            #lib_value = min(8, string_liberty_counts[string_count])
            for stone_idx in current_string:
                if current_board[stone_idx] == friendly_value:
                    friendly_lib_count[stone_idx] = lib_value
                elif current_board[stone_idx] == enemy_value:
                    enemy_lib_count[stone_idx] = lib_value

            # Now that we are done with this string, increment the string count.
            string_count += 1

    return friendly_lib_count, enemy_lib_count


def get_all_legal_moves_from_board_state(board_state):
    """
    :param board_state: a 13x13x17 board state
    :return: a 13x13 binary matrix indicating legal moves (1 = legal, 0 = illegal).
    """

    # Make a copy of the board state and reshape it.
    board_state = np.reshape(board_state.copy(), (17, 169))

    # Use this to indicate which moves are legal.
    legal_moves = [0]*169

    # The first step is probably to identify all strings at the current board state.
    black_board = board_state[14] if board_state[16][0] == 1 else board_state[15]
    white_board = board_state[14] if board_state[16][0] == 0 else board_state[15]
    current_board = get_single_storable_board_from_state(black_board, white_board)

    string_board = [-1]*169  # If a string is here, it will store the id of this string.
    string_count = 0
    liberties_board = [[] for _ in range(169)]
    string_liberty_counts = {}  # Key: string id, value: liberty count.

    for idx in range(169):
        # If this stone has already been considered, continue.
        if string_board[idx] != -1:
            continue
        # If a stone is here, it is a string.
        if current_board[idx] != 0:
            stone_value = current_board[idx]
            string_board[idx] = string_count
            string_liberty_counts[string_count] = 0
            current_string = [idx]
            for stone_idx in current_string:
                adjacent_stones = get_adjacent_intersections(stone_idx, current_board)
                for adjacent_stone in adjacent_stones:
                    adj_stone_idx = adjacent_stone[0]
                    adj_stone_value = adjacent_stone[1]
                    # If the adjacent stone value matches the strings value,
                    # and the stone has not been considered yet, add it to the string.
                    if adj_stone_value == stone_value and string_board[adj_stone_idx] == -1:
                        current_string.append(adj_stone_idx)
                        string_board[adj_stone_idx] = string_count
                    # Otherwise, if the adjacent stones value is 0, it is a liberty.
                    if adj_stone_value == 0 and string_count not in liberties_board[adj_stone_idx]:
                        string_liberty_counts[string_count] += 1
                        liberties_board[adj_stone_idx].append(string_count)

            # Now that we are done with this string, increment the string count.
            string_count += 1

    # Print the string liberties count on the board.
    """
    print("\nLiberty Counts\n")
    lib_count_print_board = []
    for string in string_board:
        if string in string_liberty_counts:
            lib_count_print_board.append(string_liberty_counts[string])
        else:
            lib_count_print_board.append(0)
    for lb_row in range(13):
        lb_row_str = ""
        for lb_col in range(13):
            lb_row_str += str(lib_count_print_board[lb_row*13 + lb_col]) + "\t"
        print(lb_row_str)
    print("\n")
    """


    # Now go back through all the intersections to find legal moves.
    playing_stone_value = 1 if board_state[16][0] == 1 else 2
    for idx in range(169):
        value_at_idx = current_board[idx]
        # If the intersection already has a stone there, the move is illegal.
        if value_at_idx != 0:
            continue
        # Get the adjacent indices.
        adjacent_stones = get_adjacent_intersections(idx, current_board)
        move_ends_in_liberties = False
        capture_count = 0
        for adjacent_stone in adjacent_stones:
            adj_stone_idx = adjacent_stone[0]
            adj_stone_value = adjacent_stone[1]
            # If an adjacent intersection is a liberty, this is a legal move.
            if adj_stone_value == 0:
                move_ends_in_liberties = True
                break
            # If an adjacent stone is a friendly string with more than one liberty, this is a legal move.
            elif adj_stone_value == playing_stone_value:
                string_id = string_board[adj_stone_idx]
                friendly_string_lib_count = string_liberty_counts[string_id]
                if friendly_string_lib_count > 1:
                    move_ends_in_liberties = True
                    break
            # If an adjacent stone is a friendly string with only one liberty, the move is legal if it ends in capture.
            #    This will be caught by one of the other adjacent stones, which must be an enemy.
            # adj_stone_value is an element of {0, 1, 2, -2}, where -2 stands for out of bounds.
            elif adj_stone_value != -2:
                # If we get here then the adjacent stone belongs to the enemy!
                string_id = string_board[adj_stone_idx]
                enemy_string_lib_count = string_liberty_counts[string_id]
                if enemy_string_lib_count <= 1:
                    # Playing here ends in the capture of the enemy string. However, this may break the Ko rule.
                    # First, check if a stone of the playing player was at this spot previously.
                    if board_state[12][idx] == 0:
                        # If a stone of this color was not previously here, the move is obviously legal.
                        move_ends_in_liberties = True
                        break
                    else:
                        # Otherwise, increment the capture count. If other stones are captured, Ko is not broken.
                        capture_count += sum([s_id == string_id for s_id in string_board])
            # If the capture count is greater than or equal to two, then the Ko rule could not possibly be broken.
            if capture_count >= 2:
                move_ends_in_liberties = True
                break

        # If playing here results in liberties, the move is legal.
        if move_ends_in_liberties:
            legal_moves[idx] = 1

    return legal_moves


def update_board_state_for_move(action_idx, board_state):
    """
    Does an update of the board state, returning the new board from the perspective of the player who will go next.
    Makes a copy, rather than inplace update.
    :param action_idx: the index that player a adds a stone at
    :param board_state: the current board state (17 x 13 x 13)
    :return: an updated board state
    """
    # Reshape the board state.
    board_state = np.reshape(board_state.copy(), (17, 169))

    # Build the current and previous boards, with 1 = friendly and -1 = enemy, given the full state.
    curr_state_friendly_player = board_state[14]
    curr_state_enemy_player = board_state[15]
    prev_state_friendly_player = board_state[12]
    prev_state_enemy_player = board_state[13]
    curr_board = [curr_state_friendly_player[idx] - curr_state_enemy_player[idx] for idx in range(169)]
    prev_board = [prev_state_friendly_player[idx] - prev_state_enemy_player[idx] for idx in range(169)]

    # Determine the outcome of the board given the next move to be made.
    move_result = board_pos_after_move(curr_board, prev_board, action_idx, 1)
    next_board = move_result["board_outcome"]
    legal_move = move_result["move_legal"]
    if not legal_move:
        print("WARNING: Illegal move made in update_board_state_for_move. Action index: ", action_idx)

    # Extract the new friendly and enemy boards from the new board state.
    new_state_friendly_player = np.zeros(169)
    new_state_enemy_player = np.zeros(169)
    for idx in range(169):
        if next_board[idx] == 1:
            new_state_friendly_player[idx] = 1
        if next_board[idx] == -1:
            new_state_enemy_player[idx] = 1

    # Create a new state, from the perspective of the enemy player. Replace oldest history values, and swap order.
    new_state = board_state.copy()
    for layer in range(2, 16, 2):
        new_state[layer - 2] = new_state[layer + 1]
        new_state[layer - 1] = new_state[layer]
    new_state[14] = new_state_enemy_player
    new_state[15] = new_state_friendly_player

    # Swap the value in the last layer, to indicate the player who will move next.
    if new_state[16][0] == 1:
        new_state[16] = [0]*169
    else:
        new_state[16] = [1]*169

    # Return the new state, organized from the perspective of the "enemy" player.
    return new_state


def print_board(black_board, white_board):
    current_line = ""
    for idx in range(169):
        if black_board[idx] == 1:
            current_line += " \U000026AB "
        elif white_board[idx] == 1:
            current_line += " \U000026AA "
        else:
            current_line += " \U00002795 "

        if (idx+1) % 13 == 0:
            print(current_line)
            current_line = ""

    print("\n\n")


def initialize_board():
    """
    Create an empty board state.
    :return: an empty board, as a 17x13x13 numpy array.
    """
    board_state = [np.zeros((13, 13)).tolist() for _ in range(16)]  # Since no moves have been made, the board should be empty.
    board_state.append(np.ones((13, 13)).tolist())
    return np.reshape(board_state, (17, 13, 13))

def save_game_to_sgf(moves_list, game_result, outfile):
    """

    :param moves_list:
    :param game_result:
    :param outfile:
    :return:
    """
    # Get the index given the alphabet values of the move.
    move_code_map = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g",
                     7: "h", 8: "i", 9: "j", 10: "k", 11: "l", 12: "m"}
    # Add the header.
    lines = ["(;GM[1]FF[4]CA[UTF-8]", "RU[Chinese]SZ[13]KM[7.5]TM[600]",
             "PW[wingobot]PB[wingobot]WR[00]BR[00]DT[2020-07-30]PC[wingo-desktop]" + "RE[" + game_result + "]GN[007]"]
    # For every move, starting with black, add it to the list of lines.
    player = "B"  # Alternates between B and W.
    for move in moves_list:
        if move == 169:
            move_code = ""  # For a pass.
        else:
            row = move // 13
            col = move % 13
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


def transform_board_and_policy(board, policy):
    """
    Rotate and flip randomly.
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




