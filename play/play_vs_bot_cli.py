"""
Script to play a game against the WinGoBot using the command line.
"""
# Python Standard
import argparse
import multiprocessing as mp
import sys

# Local
sys.path.append("..")
from source.gooop import Goban
from source.mcts_multiprocess import MonteCarloSearchTree
from source.player_controller import PlayerController


class CLIPlayer:
    """
    An object to manage game play over the CLI.
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

    def attempt_move(self, row, col):
        """
        Try making a move and get the updated board state. Reset all stones on the display.
        """
        self.go_board.make_move(row, col)
        self.go_board.print_board()

    def make_bot_move(self):
        """
        Make a move from the bots perspective.
        """
        best_bot_move = self.bot_search_tree.search(self.num_simulations)
        bot_move_row = best_bot_move // 13
        bot_move_col = best_bot_move % 13
        self.attempt_move(bot_move_row, bot_move_col)
        return best_bot_move

    @staticmethod
    def translate_human_input(human_val):
        """
        Return the human given value as an integer in [0, 13].
        """
        if human_val in [str(num) for num in range(14)]:
            return int(human_val)
        elif human_val in ["A", "a"]:
            return 10
        elif human_val in ["B", "b"]:
            return 11
        elif human_val in ["C", "c"]:
            return 12
        else:
            return 13

    def make_human_move(self):
        """
        Enter a move given human input.
        """
        # Get the user input.
        print("Row: ")
        row = self.translate_human_input(input())
        print("Col: ")
        col = self.translate_human_input(input())
        # Make the move.
        human_move = row * 13 + col
        self.attempt_move(row, col)
        # Update the bot's search tree to account for the human move.
        if human_move in self.bot_search_tree.root.children:
            self.bot_search_tree.update_root(self.bot_search_tree.root.children[human_move])
        else:
            self.bot_search_tree = MonteCarloSearchTree(0, self.bot_queues, self.go_board.copy())
        return human_move

    def start_game(self):
        """
        Play a full game.
        """
        # Play until the move limit is reached or two consecutive passes are made.
        consecutive_passes = 0
        for move_num in range(self.game_len // 2):
            # Black makes a move.
            if self.bot_color == "B":
                move = self.make_bot_move()
            else:
                move = self.make_human_move()
            # Check if we can exit early.
            if move == self.pass_val:
                consecutive_passes += 1
                if consecutive_passes == 2:
                    break
            else:
                consecutive_passes = 0
            # White makes a move.
            if self.bot_color == "W":
                move = self.make_bot_move()
            else:
                move = self.make_human_move()
            # Check if we can exit early.
            if move == self.pass_val:
                consecutive_passes += 1
                if consecutive_passes == 2:
                    break
            else:
                consecutive_passes = 0
        # Score the game and print the winner.
        game_outcome = self.go_board.ogs_score()
        print("\n\nGood Game. Result: {}".format(game_outcome))

    def save_game(self, outfile):
        """
        Save the game to an SGF file.
        """
        self.go_board.save_game_to_sgf(outfile)

    def shutdown(self):
        """
        Close the processes that were spun up for the player controller.
        """
        self.bot_player_controller.shutdown()


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
    cli_player = CLIPlayer(args.length, bot_color, args.simulations, args.botfile)
    cli_player.start_game()

    # When the game ends, save it and shut the bot down.
    cli_player.save_game(args.outfile)
    cli_player.shutdown()


if __name__ == "__main__":
    main()