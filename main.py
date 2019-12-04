import tensorflow as tf
import os
from modules.GA import GA
from modules.NN import construct_network
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if tf.test.is_gpu_available():
    print("Using GPU: {}".format(tf.test.gpu_device_name()))
else:
    print("No GPU detected, using CPU mode.")

ga = GA(20, 5, 10)
ga.construct_networks()
print(ga.networks)