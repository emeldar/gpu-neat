from modules.gene import Gene
import numpy as np


class Genome:
    def __init__(self, num_in, num_out, ga):
        self.num_in = num_in
        self.num_out = num_out
        self.genes = []
        self.ga = ga
        for i in range(num_in):
            for j in range(num_out):
                self.genes.append(Gene(str(i), str(1000+j), np.random.normal(0, 1), True, self.ga.get_innov()))
