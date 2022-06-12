"""
This utility will display a game from the H5 file in MatPlotLib.
The display is interactive (left click to see the previous state, right click to see the next).
"""

# Python Standard
import argparse
import math
import random
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


class BoardDisplay:
    """
    Instances of this class display a game on a Matplotlib subplot.
    """
    def __init__(self, fig, ax, h5_file, game_number):
        """
        Load the moves from the given h5 file / game number and plot them on the axis.
        """
        # Initialize the Goban.
        self.go_board = Goban(13)

        # Build the empty board.
        self.fig = fig
        self.ax = ax
        self.build_empty_board()

        # Load the moves from this game.
        self.moves_list = []
        self.load_moves_from_game(h5_file, game_number)
        self.move_num = len(self.moves_list) - 1
        #self.move_num = 32

        # Connect click events with this objects callback method.
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def build_empty_board(self):
        self.ax.set_facecolor('xkcd:sandy')
        self.ax.axis('equal')
        self.ax.set(xlim=(0, 12), ylim=(0, 12))
        self.ax.set_xticks(np.arange(13))
        self.ax.set_yticks(np.arange(13))
        self.ax.grid(color="xkcd:black", linewidth=1)
        self.ax.set_axisbelow(True)
        self.ax.set_aspect('equal', 'box')

    def reset_plot(self):
        self.ax.clear()
        self.ax.set_facecolor('xkcd:sandy')
        self.ax.axis('equal')
        self.ax.set(xlim=(0, 12), ylim=(0, 12))
        self.ax.set_xticks(np.arange(13))
        self.ax.set_yticks(np.arange(13))
        self.ax.grid(color="xkcd:black", linewidth=1)
        self.ax.set_axisbelow(True)
        self.ax.set_aspect('equal', 'box')

    def set_stone_of_color(self, row, col, stone_color, str_id="EMPTY", lib_cnt=0, bl="", wl="", al=""):
        """
        Draw a stone of a particular color on the board at the given spot.
        """
        if str_id == "EMPTY":
            stone = patches.Circle([col, row], radius=0.25, facecolor="xkcd:sandy")
            str_to_display = bl + "\n" + wl + "\n" + al
            #self.ax.annotate(str_to_display, (col, row), color='xkcd:black', weight='bold',
            #            fontsize=6, ha='center', va='center')
        elif stone_color == "B":
            stone = patches.Circle([col, row], radius=0.5, facecolor="xkcd:black")
            str_to_display = str_id + "\n" + lib_cnt
            #self.ax.annotate(str_to_display, (col, row), color='xkcd:white', weight='bold',
            #            fontsize=6, ha='center', va='center')
        else:
            stone = patches.Circle([col, row], radius=0.5, facecolor="xkcd:white")
            str_to_display = str_id + "\n" + lib_cnt
            #self.ax.annotate(str_to_display, (col, row), color='xkcd:black', weight='bold',
            #            fontsize=6, ha='center', va='center')
        self.ax.add_patch(stone)

    def attempt_move(self, row, col, update=False):
        """
        Try making a move and get the updated board state. Reset all stones on the display.
        """
        adjusted_row = 12 - row
        self.go_board.make_move(adjusted_row, col)
        if not update:
            return
        self.reset_plot()
        current_board = self.go_board.get_current_board()
        current_str_ids = self.go_board.get_current_str_ids()
        curr_lib_counts, added_libs = self.go_board.get_current_lib_counts()
        black_legal = self.go_board.legal_black_moves
        white_legal = self.go_board.legal_white_moves
        for s_row in range(13):
            for s_col in range(13):
                stone_val = current_board[s_row, s_col]
                str_id = current_str_ids[s_row, s_col]
                lib_count = str(curr_lib_counts[s_row, s_col])
                bl = "BL" if black_legal[s_row, s_col] == 1 else ""
                wl = "WL" if white_legal[s_row, s_col] == 1 else ""
                al = str(added_libs[s_row, s_col])
                if stone_val == self.go_board.white:
                    self.set_stone_of_color(12 - s_row, s_col, "W", str_id=str_id, lib_cnt=lib_count, bl=bl, wl=wl, al=al)
                elif stone_val == self.go_board.black:
                    self.set_stone_of_color(12 - s_row, s_col, "B", str_id=str_id, lib_cnt=lib_count, bl=bl, wl=wl, al=al)
                else:
                    self.set_stone_of_color(12 - s_row, s_col, "E", str_id=str_id, lib_cnt=lib_count, bl=bl, wl=wl, al=al)

    def display_up_to(self, move_number):
        self.go_board = Goban(13)
        move_total_time = 0.0
        for idx in range(move_number):
            row, col = self.moves_list[idx]
            start = time.time()
            self.attempt_move(row, col)
            end = time.time()
            move_time = end - start
            move_total_time += move_time
        row, col = self.moves_list[move_number]
        self.attempt_move(row, col, update=True)

    def onclick(self, event):
        """
        Whenever the user clicks, load and show the next move.
        """
        if event.button == 1:
            self.move_num -= 1
            self.display_up_to(self.move_num)
        elif event.button == 3:
            self.move_num += 1
            self.display_up_to(self.move_num)

    def load_moves_from_game(self, h5_file, game_number):
        h5_data = h5py.File(h5_file, "r")
        game_data = h5_data["games"]["game_" + str(game_number)]
        self.moves_list = game_data["moves"][()].tolist()


def get_num_games_from_h5(h5_file):
    h5_data = h5py.File(h5_file, "r")
    num_games = len(h5_data["games"])
    return num_games


def main():
    # Parse input args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="../self_play_games/h5_games/batch_1608.h5")
    parser.add_argument("-r", "--rows", default=4, type=int)
    parser.add_argument("-c", "--cols", default=4, type=int)
    args = parser.parse_args()
    h5_file = args.h5
    rows = args.rows
    cols = args.cols
    total_games_available = get_num_games_from_h5(h5_file)

    # Create a display for however many random games are requested.
    fig, axes = plt.subplots(rows, cols)
    displays = []
    for row in range(rows):
        for col in range(cols):
            ax = axes[row, col]
            # Pick a random game number.
            game_num = random.choice(range(total_games_available))
            # Create a display foir that game.
            new_display = BoardDisplay(fig, ax, h5_file, game_num)
            new_display.display_up_to(new_display.move_num)
            displays.append(new_display)

    # Display the plot.
    plt.show()


if __name__ == "__main__":
    main()
