def tromp_taylor_score(board):
    """
    Scores a game using Tromp Taylor rules: https://codegolf.stackexchange.com/questions/6693/score-a-game-of-go
    Basically, process every empty space as a string.
    If this string is only surrounded by one player (and the edge of the board), it is that player's territory.
    :param board: a 19x19 matrix representing board state
    :return: the value of the winning player's stone (-1 or 1)
    """
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
        curr_player_board = [1 if x is curr_player_value else 0 for x in byte_board]
        next_player_board = [1 if x is next_player_value else 0 for x in byte_board]
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
    string_board = [-1]*169  # If a string is here, it will store the id of this string.
    string_count = 0
    string_liberty_counts = {}  # Key: string id, value: liberty count.A
    liberties_board = [[]]*169
    friendly_lib_count = [0]*169
    enemy_lib_count = [0]*169

    for idx in range(169):
        # If this stone has already been considered, continue.
        if string_board[idx] != -1:
            continue
        # If a stone is here, it is a string.
        if current_board[idx] is not 0:
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
            # Set the value of the liberties for this string.
            for stone_idx in current_string:
                if current_board[stone_idx] == friendly_value:
                    friendly_lib_count[stone_idx] = string_liberty_counts[string_count]
                else:
                    enemy_lib_count[stone_idx] = string_liberty_counts[string_count]

            # Now that we are done with this string, increment the string count.
            string_count += 1


def get_all_legal_moves_from_board_state(board_state):
    """
    :param board_state: a 13x13x17 board state
    :return: a 13x13 binary matrix indicating legal moves (1 = legal, 0 = illegal).
    """

    # Use this to indicate which moves are legal.
    legal_moves = [0]*169

    # The first step is probably to identify all strings at the current board state.
    black_board = board_state[14] if board_state[16][0] == 1 else board_state[15]
    white_board = board_state[14] if board_state[16][0] == 0 else board_state[15]
    current_board = get_single_storable_board_from_state(black_board, white_board)
    string_board = [-1]*169  # If a string is here, it will store the id of this string.
    string_count = 0
    liberties_board = [[]]*169
    string_liberty_counts = {}  # Key: string id, value: liberty count.

    for idx in range(169):
        # If this stone has already been considered, continue.
        if string_board[idx] != -1:
            continue
        # If a stone is here, it is a string.
        if current_board[idx] is not 0:
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
                        liberties_board[adjacent_stone_idx].append(string_count)

            # Now that we are done with this string, increment the string count.
            string_count += 1

    # Now go back through all the intersections to find legal moves.
    playing_stone_value = 1 if board_state[16][0] == 1 else 2
    for idx in range(169):
        value_at_idx = current_board[idx]
        # If the intersection already has a stone there, the move is illegal.
        if value_at_idx is not 0:
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
            # If an adjaent stone is a friendly string with only one liberty, the move is legal bc it ends in capture.
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
                        capture_count += 1
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
    Does an inplace update of the board state.
    :param action_idx: the index that player a adds a stone at
    :param board_state: the current board state (13 x 13 x 17)
    :return: an updated board state
    """

    # Determine the outcome of the board given the next move to be made.
    curr_state_curr_player = board_state[14]
    curr_state_next_player = board_state[15]
    prev_state_curr_player = board_state[12]
    prev_state_next_player = board_state[13]
    curr_board = [curr_state_curr_player[idx] - curr_state_next_player[idx] for idx in range(169)]
    prev_board = [prev_state_curr_player[idx] - prev_state_next_player[idx] for idx in range(169)]
    next_board = board_pos_after_move(curr_board, prev_board, action_idx, 1)["board_outcome"]
    new_state_curr_player = [0]*169
    new_state_next_player = [0]*169
    for idx in range(169):
        if next_board[idx] == 1:
            new_state_curr_player[idx] = 1
        if next_board[idx] == -1:
            new_state_next_player[idx] = 1

    # Replace oldest history value by shifting down and swapping order for next player.
    for layer in range(2, 16, 2):
        board_state[layer - 2] = board_state[layer + 1]
        board_state[layer - 1] = board_state[layer]

    board_state[14] = new_state_next_player
    board_state[15] = new_state_curr_player

    # Swap the value in the last layer.
    if board_state[16][0] == 1:
        board_state[16] = [0]*169
    else:
        board_state[16] = [1]*169

    return board_state


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






