from threading import Thread
from nn_ll_tf import PolicyValueNetwork as pvn
import numpy as np
import tensorflow as tf

def tfunc():
    from nn_ll_tf import PolicyValueNetwork as pvn
    train_data_input = np.ones((1024, 19, 13, 13))
    train_data_p = np.zeros((1024, 170))
    train_data_v = np.ones((1024))
    bot = pvn(0.001, train_reinforcement=True, starting_network_file="shodan_fossa/shodan_focal_fossa_92.h5")
    bot.train_supervised(train_data_input, train_data_v, train_data_p)
    fake_input = np.random.random((128, 19, 13, 13))
    output = bot.predict_w_trt(fake_input)
    policy = output[0]["policy"]
    print("Type of policy: ", type(policy))

t1 = Thread(target=tfunc)
t2 = Thread(target=tfunc)

def main():
    t1.start()
    t2.start()
    t1.join()
    t2.join()

main()
