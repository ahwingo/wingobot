"""
This utility runs a speed test of the main Policy value NN.
"""
# Python Standard
import sys
import time

# Third Party
from matplotlib import pyplot as plt
import numpy as np

# Local
sys.path.append("..")
from source.nn_ll_tf import PolicyValueNetwork as PVN


def main():
    # Load the bot.
    #bot = PVN(0.001, starting_network_file="../misc_models/shodan_fossa/shodan_focal_fossa_1.h5", trt_mode=True)
    bot = PVN(0.001, trt_mode=False)

    # Record how many states per second the bot can process.
    states_per_second = []
    batch_sizes = [2**x for x in range(13)]
    reps = 4

    # Initialize random states.
    for batch_size in batch_sizes:
        state = np.random.random((batch_size, 19, 13, 13))
        start = time.time()
        for _ in range(reps):
            bot.predict_given_state(state, batch_size=batch_size)
        end = time.time()
        total_time = end - start
        throughput = reps * batch_size / total_time
        states_per_second.append(throughput)

        # Estimate a few stats on how many games could be played a day.
        time_per_long_game_batch = total_time * 128 * 64 / (reps * 3600)
        games_per_day = 24.0 * batch_size / time_per_long_game_batch
        print("Batch size {}. Throughput {}. Time / game batch {}. Games / day {}. Secs / state {}.".format(batch_size,
                                                                                              throughput,
                                                                                              time_per_long_game_batch,
                                                                                              games_per_day,
                                                                                              total_time / (reps)))

    # Plot the results.
    plt.plot(np.asarray(batch_sizes), np.asarray(states_per_second))
    plt.xlabel("Batch Size")
    plt.ylabel("States Evaluated per Second")
    plt.title("Effect of Batch Processing on Model Throughput")
    plt.savefig("nn_batch_speed_test_results.png")


if __name__ == "__main__":
    main()