"""
Script to play a game against the WinGoBot using MatPlotLib as the GUI.
"""
# Python Standard
import argparse
import multiprocessing as mp
import sys

# Third Party
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Local
sys.path.append("..")
from source.gooop import Goban
from source.mcts_multiprocess import MonteCarloSearchTree
from source.player_controller import PlayerController


class MPLPlayer:
    """
    An object to manage game play over a matplotlib GUI.
    """
    def __init__(self, game_len, bot_color, num_simulations, bot_file):
        # Store user set params.
        self.game_len = game_len
        self.bot_color = bot_color
        self.num_simulations = num_simulations

        # Initialize the board.
        goban_size = 13
        self.go_board = Goban(goban_size)
        self.pass_val = goban_size**2

        # Initialize the player controller queues.
        self.queue_manager = mp.Manager()
        self.bot_input_queue = self.queue_manager.Queue()
        self.bot_output_queue = self.queue_manager.Queue()
        self.bot_output_queues_map = {0: self.bot_output_queue}
        self.bot_player_controller = PlayerController(bot_file, 1, self.bot_input_queue)
        self.bot_player_controller.start()
        self.bot_player_controller.set_output_queue_map(self.bot_output_queues_map)
        self.bot_queues = {"input": self.bot_input_queue, "output": self.bot_output_queue}

        # Initialize the bots search tree.
        self.bot_search_tree = MonteCarloSearchTree(0, self.bot_queues, self.go_board)

        # Keep track of the move number (can be used to derive current player).
        self.total_moves = 0

        # Create the GUI board.
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('xkcd:sandy')
        self.ax.axis('equal')
        self.ax.set(xlim=(0, 12), ylim=(0, 12))
        self.ax.set_xticks(np.arange(13))
        self.ax.set_yticks(np.arange(13))
        self.ax.grid(color="xkcd:black", linewidth=1)
        self.ax.set_axisbelow(True)
        self.ax.set_aspect('equal', 'box')

        # Create the user click callback.
        def on_click(event):
            """
            If the user clicks on an intersection (within 20 %)
            add a stone to that spot if its a legal move.
            """
            # Set thresholds.
            threshold_low = 0.4
            threshold_high = 1.0 - threshold_low
            # Extract event data.
            x_pos = int(event.xdata)
            x_extra = event.xdata - x_pos
            y_pos = int(event.ydata)
            y_extra = event.ydata - y_pos
            # Determine the corresponding rows and columns.
            valid_click = False
            if x_extra < threshold_low and y_extra < threshold_low:
                row = 12 - y_pos
                col = x_pos
                valid_click = True
            elif x_extra < threshold_low and y_extra > threshold_high:
                row = 12 - y_pos - 1
                col = x_pos
                valid_click = True
            elif x_extra > threshold_high and y_extra < threshold_low:
                row = 12 - y_pos
                col = x_pos + 1
                valid_click = True
            elif x_extra > threshold_high and y_extra > threshold_high:
                row = 12 - y_pos - 1
                col = x_pos + 1
                valid_click = True
            # Make the human move.
            if valid_click:
                self.make_human_move(row, col)
                human_move = row * 13 + col
                # Make a bot move, if there are still moves to be made.
                if self.total_moves < self.game_len:
                    self.make_bot_move(human_move=human_move)
        self.fig.canvas.mpl_connect('button_press_event', on_click)

    def make_human_move(self, row, col):
        print(row, col)
        self.attempt_move(row, col)

    def make_bot_move(self, human_move=None):
        # If a human has recently made a move, update the bot tree.
        if human_move is not None and human_move in self.bot_search_tree.root.children:
            self.bot_search_tree.update_root(self.bot_search_tree.root.children[human_move])
        elif human_move is not None:
            self.bot_search_tree = MonteCarloSearchTree(0, self.bot_queues, self.go_board.copy())
        # Make a bot move.
        best_bot_move = self.bot_search_tree.search(self.num_simulations)
        bot_move_row = best_bot_move // 13
        bot_move_col = best_bot_move % 13
        self.attempt_move(bot_move_row, bot_move_col)

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

    def set_stone_of_color(self, row, col, stone_color):
        """
        Draw a stone of a particular color on the board at the given spot.
        """
        if stone_color == "B":
            stone = patches.Circle([col, row], radius=0.5, facecolor="xkcd:black")
        else:
            stone = patches.Circle([col, row], radius=0.5, facecolor="xkcd:white")
        self.ax.add_patch(stone)

    def attempt_move(self, row, col):
        """
        Try making a move and get the updated board state. Reset all stones on the display.
        """
        # Make the move.
        self.reset_plot()
        self.go_board.make_move(row, col)
        current_board = self.go_board.get_current_board()
        for s_row in range(13):
            for s_col in range(13):
                stone_val = current_board[s_row, s_col]
                if stone_val == self.go_board.white:
                    self.set_stone_of_color(12 - s_row, s_col, "W")
                elif stone_val == self.go_board.black:
                    self.set_stone_of_color(12 - s_row, s_col, "B")
        # Increment the count of moves made.
        self.total_moves += 1
        # If the move count has been reached, shut things down.
        if self.total_moves == self.game_len:
            self.shutdown()

    def start_game(self):
        """
        Play a full game.
        """
        plt.show()

    def shutdown(self):
        """
        Close the GUI and shut down the player.
        """
        plt.close()
        self.bot_player_controller.shutdown()

    def save_game(self, outfile):
        """
        Save the game to an SGF file.
        """
        self.go_board.save_game_to_sgf(outfile)

def main():
    # Set some game params.
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", default=128, type=int,
                        help="The length of the game (total number of moves to be made).")
    parser.add_argument("-s", "--simulations", default=32, type=int,
                        help="The number of simulations the bot will make before each move.")
    parser.add_argument("-c", "--color", default="W",
                        help="W or B. The human player's stone color.")
    parser.add_argument("-o", "--outfile", default="human_vs_bot.sgf",
                        help="The path to which the game should be saved to.")
    parser.add_argument("-b", "--botfile", default="../models/shodan_focal_fossa_161.h5",
                        help="The path to WinGoBot weights file to initialize the computer player with.")
    args = parser.parse_args()

    # Create and start a player.
    bot_color = "B" if args.color == "W" else "W"
    mpl_player = MPLPlayer(args.length, bot_color, args.simulations, args.botfile)
    mpl_player.start_game()

    # When the game ends, save it and shut the bot down.
    mpl_player.save_game(args.outfile)
    mpl_player.shutdown()


if __name__ == "__main__":
    main()
