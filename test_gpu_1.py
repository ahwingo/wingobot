import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from nn_ll_tf import PolicyValueNetwork
network_file = "young_dark_rock_ckpt_12000.h5"
network = PolicyValueNetwork(0.0001, starting_network_file=network_file)
time.sleep(20)
