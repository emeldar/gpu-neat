from modules.genome import Genome
from modules.NN import construct_network

class GA:
    def __init__(self, num_in, num_out, pop_size):
        self.global_inv = 0
        self.num_in = num_in
        self.num_out = num_out
        self.pop_size = pop_size
        self.genomes = []
        self.networks = []
        for i in range(pop_size):
            self.genomes.append(Genome(num_in, num_out, self))

    def construct_networks(self):
        self.networks = []
        for genome in self.genomes:
            self.networks.append(construct_network(genome))

    def get_innov(self):
        self.global_inv += 1
        return self.global_inv - 1

    def get_genomes(self):
        return self.genomes
