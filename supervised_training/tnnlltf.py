"""
Train the WinGoBot through supervised learning on a set of games, stored in h5 files.
Runs in an infinite loop until shutdown.
"""

# Python Standard
import argparse
import datetime
import os
import sys

# Local / Third Party
sys.path.append("..")
from source.nn_ll_tf import *
from source.training_library import TrainingLibrary
import numpy as np
import tensorflow as tf


def optimize_w_generator(player_nn, model_name, trainer, batch_size, output_dir):
    training_files = trainer.registered_h5_files

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(output_dir + "/{epoch}_{policy_categorical_accuracy:.3f}.h5",
                                                             monitor='policy_categorical_accuracy',
                                                             verbose=0,
                                                             save_best_only=True,
                                                             save_weights_only=False,
                                                             mode='auto',
                                                             save_freq='epoch')

    def gen():
        inputs, gt_policies, gt_values = trainer.get_random_training_batch(training_files, batch_size, 8, 13)
        yield inputs, {"value": gt_values, "policy": gt_policies}

    dataset = tf.data.Dataset.from_generator(gen, (np.int8, {"value": np.int8, "policy": np.int8})).repeat()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    player_nn.train_supervised_gen(dataset, batch_size=batch_size, callbacks=[tensorboard_callback, checkpoint_callback])


def optimization_loop(player_nn, model_name, trainer, batch_size, output_dir):
    mini_batch_num = 0
    training_files = trainer.registered_h5_files

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    while True:
        inputs, gt_policies, gt_values = trainer.get_random_training_batch_mp(training_files, batch_size, 8, 13)
        print("Training on mini batch ", mini_batch_num)
        history = player_nn.train_supervised(inputs, gt_values, gt_policies, callbacks=[tensorboard_callback])
        print(history)
        mini_batch_num += 1

        # Save a checkpoint every 500 mini batches.
        if mini_batch_num % 500 == 0:
            ckpt_file = model_name + "_ckpt_" + str(mini_batch_num) + ".h5"
            player_nn.save_checkpoint(os.path.join(output_dir, ckpt_file))


def main():
    # Parse input args.
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot_name", default="wingobot_sl")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--data_dir", default="original_online_games/h5_jun10")
    parser.add_argument("--output_dir", default="wingobot_sl_jun_10_2022")
    args = parser.parse_args()

    # Create the output directory if it does not exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Create the player.
    player_nn = PolicyValueNetwork(0.0001, train_supervised=True, starting_network_file="wingobot_sl_jun2022/wingobot_sl_ckpt_16500.h5")
    #player_nn = PolicyValueNetwork(0.0001, train_supervised=True)
    player_nn.save_checkpoint(os.path.join(args.output_dir, args.bot_name + ".h5"))

    # Create a training library, which gets access to all of the self play h5 files.
    trainer = TrainingLibrary()
    training_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith("h5")]
    for h5_file in training_files:
        trainer.register_h5_file(h5_file)

    # Run the optimization loop on this thread.
    optimize_w_generator(player_nn, args.bot_name, trainer, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
