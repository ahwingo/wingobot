from neural_network_tiny import *
import random
import time


# Generate 8 random board positions.



network = PolicyValueNetwork(0.01)
one_sum = 0.0
eight_sum = 0.0
thirty_two_sum = 0.0
sixty_four_sum = 0.0
for trial in range(600):
    one_board_state = np.random.rand(1, 13, 13, 9)
    eight_board_states = np.random.rand(8, 13, 13, 9)
    thirty_two_board_states = np.random.rand(32, 13, 13, 9)
    sixty_four_board_states = np.random.rand(64, 13, 13, 9)

    start_time = time.time()
    network.predict_given_state(one_board_state, 1)
    end_time = time.time()
    time_diff = end_time - start_time
    one_sum += time_diff

    start_time = time.time()
    network.predict_given_state(eight_board_states, 8)
    end_time = time.time()
    time_diff = end_time - start_time
    eight_sum += time_diff

    start_time = time.time()
    network.predict_given_state(thirty_two_board_states, 32)
    end_time = time.time()
    time_diff = end_time - start_time
    thirty_two_sum += time_diff

    start_time = time.time()
    network.predict_given_state(sixty_four_board_states, 64)
    end_time = time.time()
    time_diff = end_time - start_time
    sixty_four_sum += time_diff


print("One average = " + str(one_sum / 600.0))
print("Eight average = " + str(eight_sum / 600.0))
print("Thirty two average = " + str(thirty_two_sum / 600.0))
print("Sixty four average = " + str(sixty_four_sum / 600.0))
