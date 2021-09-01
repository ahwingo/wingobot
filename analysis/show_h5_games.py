"""
This utility will display a game from the H5 file in MatPlotLib.
The display is interactive (left click to see the previous state, right click to see the next).
"""

# Python Standard
import argparse
import sys
import time

# Third Party
import h5py
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Local
sys.path.append("..")
from source.gooop import Goban

# Initialize the Goban.
go_board = Goban(13)

# Build the empty board.
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_facecolor('xkcd:sandy')
ax.axis('equal')
ax.set(xlim=(0, 12), ylim=(0, 12))
ax.set_xticks(np.arange(13))
ax.set_yticks(np.arange(13))
ax.grid(color="xkcd:black", linewidth=1)
ax.set_axisbelow(True)
ax.set_aspect('equal', 'box')

# Use this to keep track of the moves list.
moves_list = []
move_num = 122


def reset_plot():
    ax.clear()
    ax.set_facecolor('xkcd:sandy')
    ax.axis('equal')
    ax.set(xlim=(0, 12), ylim=(0, 12))
    ax.set_xticks(np.arange(13))
    ax.set_yticks(np.arange(13))
    ax.grid(color="xkcd:black", linewidth=1)
    ax.set_axisbelow(True)
    ax.set_aspect('equal', 'box')


def set_stone_of_color(row, col, stone_color, str_id="EMPTY", lib_cnt=0, bl="", wl="", al=""):
    """
    Draw a stone of a particular color on the board at the given spot.
    """
    if str_id == "EMPTY":
        stone = patches.Circle([col, row], radius=0.25, facecolor="xkcd:sandy")
        str_to_display = bl + "\n" + wl + "\n" + al
        ax.annotate(str_to_display, (col, row), color='xkcd:black', weight='bold',
                    fontsize=6, ha='center', va='center')
    elif stone_color == "B":
        stone = patches.Circle([col, row], radius=0.5, facecolor="xkcd:black")
        str_to_display = str_id + "\n" + lib_cnt
        ax.annotate(str_to_display, (col, row), color='xkcd:white', weight='bold',
                    fontsize=6, ha='center', va='center')
    else:
        stone = patches.Circle([col, row], radius=0.5, facecolor="xkcd:white")
        str_to_display = str_id + "\n" + lib_cnt
        ax.annotate(str_to_display, (col, row), color='xkcd:black', weight='bold',
                    fontsize=6, ha='center', va='center')
    ax.add_patch(stone)


def attempt_move(row, col, update=False):
    """
    Try making a move and get the updated board state. Reset all stones on the display.
    """
    adjusted_row = 12 - row
    go_board.make_move(adjusted_row, col)
    if not update:
        return
    reset_plot()
    current_board = go_board.get_current_board()
    current_str_ids = go_board.get_current_str_ids()
    curr_lib_counts, added_libs = go_board.get_current_lib_counts()
    black_legal = go_board.legal_black_moves
    white_legal = go_board.legal_white_moves
    for s_row in range(13):
        for s_col in range(13):
            stone_val = current_board[s_row, s_col]
            str_id = current_str_ids[s_row, s_col]
            lib_count = str(curr_lib_counts[s_row, s_col])
            bl = "BL" if black_legal[s_row, s_col] == 1 else ""
            wl = "WL" if white_legal[s_row, s_col] == 1 else ""
            al = str(added_libs[s_row, s_col])
            if stone_val == go_board.white:
                set_stone_of_color(12 - s_row, s_col, "W", str_id=str_id, lib_cnt=lib_count, bl=bl, wl=wl, al=al)
            elif stone_val == go_board.black:
                set_stone_of_color(12 - s_row, s_col, "B", str_id=str_id, lib_cnt=lib_count, bl=bl, wl=wl, al=al)
            else:
                set_stone_of_color(12 - s_row, s_col, "E", str_id=str_id, lib_cnt=lib_count, bl=bl, wl=wl, al=al)


def onclick_fast(event):
    """
    Whenever the user clicks, load and show the next move.
    """
    global move_num
    for _ in moves_list[:-1]:
        row, col = moves_list[move_num]
        attempt_move(row, col)
        move_num += 1
    row, col = moves_list[move_num]
    attempt_move(row, col, update=True)

def display_up_to(move_number):
    global go_board
    go_board = Goban(13)
    move_total_time = 0.0
    for idx in range(move_number):
        row, col = moves_list[idx]
        start = time.time()
        attempt_move(row, col)
        end = time.time()
        move_time = end - start
        move_total_time += move_time
    print("All moves took {} seconds.".format(move_total_time))
    row, col = moves_list[move_number]
    start = time.time()
    score = go_board.tromp_taylor_score()
    end = time.time()
    print("Scoring took {} seconds. Score = {}.".format((end - start), score))
    attempt_move(row, col, update=True)

def onclick(event):
    """
    Whenever the user clicks, load and show the next move.
    """
    global move_num
    if event.button == 1:
        move_num -= 1
        display_up_to(move_num)
        print("here")
    elif event.button == 3:
        move_num += 1
        display_up_to(move_num)

def load_moves_from_game(h5_file, game_number):
    """

    """
    global moves_list
    h5_data = h5py.File(h5_file, "r")
    game_data = h5_data["games"]["game_" + str(game_number)]
    moves_list = game_data["moves"][()].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5")
    parser.add_argument("--game")
    args = parser.parse_args()
    load_moves_from_game(args.h5, args.game)

    # Display the plot. Start looping.
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
