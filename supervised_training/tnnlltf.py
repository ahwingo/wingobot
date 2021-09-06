"""
Train the WinGoBot through supervised learning on a set of games, stored in h5 files.
Runs in an infinite loop until shutdown.
"""

# Python Standard
import argparse
import os
import sys

# Local
sys.path.append("..")
from source.nn_ll_tf import *
from source.training_library import TrainingLibrary


def optimization_loop(player_nn, model_name, trainer, batch_size, output_dir):
    mini_batch_num = 0
    training_files = trainer.registered_h5_files
    while True:
        inputs, gt_policies, gt_values = trainer.get_random_training_batch(training_files, batch_size, 8, 13)
        print("Training on mini batch ", mini_batch_num)
        player_nn.train_supervised(inputs, gt_values, gt_policies)
        mini_batch_num += 1

        # Save a checkpoint every 500 mini batches.
        if mini_batch_num % 500 == 0:
            ckpt_file = model_name + "_ckpt_" + str(mini_batch_num) + ".h5"
            player_nn.save_checkpoint(os.path.join(output_dir, ckpt_file))


def main():
    # Parse input args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot_name", default="wingobot_sl")
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--data_dir", default="../self_play_games/h5_games")
    parser.add_argument("--output_dir", default="wingobot_sl")
    args = parser.parse_args()

    # Create the output directory if it does not exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Create the player.
    player_nn = PolicyValueNetwork(0.0001, train_supervised=True)
    player_nn.save_checkpoint(os.path.join(args.output_dir, args.bot_name + ".h5"))

    # Create a training library, which gets access to all of the self play h5 files.
    trainer = TrainingLibrary()
    training_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith("h5")]
    for h5_file in training_files:
        trainer.register_h5_file(h5_file)

    # Run the optimization loop on this thread.
    optimization_loop(player_nn, args.bot_name, trainer, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
