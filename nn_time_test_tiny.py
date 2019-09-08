from neural_network_tiny import *
import random
import time


# Generate 8 random board positions.



network = PolicyValueNetwork(0.01)
eight_sum = 0.0
one_sum = 0.0
for trial in range(600):
    eight_board_states = np.random.rand(64, 13, 13, 9)
    one_board_state = np.random.rand(1, 13, 13, 9)
    start_time = time.time()
    network.predict_given_state(eight_board_states, 64)
    end_time = time.time()
    time_diff = end_time - start_time
    #print("Eight states predicted in " + str(time_diff) + " seconds.")
    eight_sum += time_diff

    start_time = time.time()
    network.predict_given_state(one_board_state, 1)
    end_time = time.time()
    time_diff = end_time - start_time
    #print("One states predicted in " + str(time_diff) + " seconds.")
    one_sum += time_diff


print("Sixty four average = " + str(eight_sum / 600.0))
print("One average = " + str(one_sum / 600.0))
