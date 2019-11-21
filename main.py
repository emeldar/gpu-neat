import tensorflow as tf
import os
from modules.GA import GA
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if tf.test.is_gpu_available():
    print("Using GPU: {}".format(tf.test.gpu_device_name()))
else:
    print("No GPU detected, using CPU mode.")

ga = GA(3, 2, 2)
for genome in ga.get_genomes():
    for gene in genome.genes:
        print(gene.input, gene.output)