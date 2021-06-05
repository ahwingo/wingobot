import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from nn_ll_tf import PolicyValueNetwork as PVN
#from nnlltf_cpu import PolicyValueNetwork as PVNCPU
import time

#bot = PVN(0.001, starting_network_file="shodan_fossa/shodan_focal_fossa_1.h5", trt_mode=True)
bot = PVN(0.001, trt_mode=False)
#bot = PVNCPU(0.001, trt_mode=False)

states_per_second = []
batch_sizes = [2**x for x in range(13)]
reps = 4

for batch_size in batch_sizes:
    #state = np.ones((batch_size, 19, 13, 13), dtype=np.int8)
    state = np.random.random((batch_size, 19, 13, 13))
    #state = np.random.random((batch_size, 13, 13, 19))
    start = time.time()
    for _ in range(reps):
        bot.predict_given_state(state, batch_size=batch_size)
    end = time.time()
    total_time = end - start
    throughput = reps * batch_size / total_time
    time_per_long_game_batch = total_time * 128 * 64 / (reps * 3600)
    games_per_day = 24.0 * batch_size / time_per_long_game_batch
    print("Batch size {}. Throughput {}. Time per game batch {}. Games per day {}. Secs per state {}.".format(batch_size,
                                                                                          throughput,
                                                                                          time_per_long_game_batch,
                                                                                          games_per_day,
                                                                                          total_time / (reps)))
    states_per_second.append(throughput)


print(states_per_second)
