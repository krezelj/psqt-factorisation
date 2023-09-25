from specimens import Specimen
import numpy as np
from scipy.special import softmax
from typing import List

N_PARENTS_IF_CROSSOVER = 2


class Evolution():

    __slots__ = ['specimen_type',
                 'all_specimens',
                 'generation_size',
                 'top_k',
                 'init_args',
                 'mutation_args',
                 'allow_crossover',
                 'allow_mutation',
                 'favor_best',
                 'last_fitness',
                 'fitness_history']

    def __init__(self,
                 specimen_type,
                 generation_size,
                 top_k,
                 init_args={},
                 mutation_args={},
                 allow_crossover=True,
                 allow_mutation=True,
                 favor_best=True,
                 ) -> None:
        assert (top_k <= generation_size)
        assert (allow_crossover or allow_mutation)
        self.specimen_type = specimen_type
        self.generation_size = generation_size
        self.init_args = init_args
        self.mutation_args = mutation_args
        self.top_k = top_k
        self.allow_crossover = allow_crossover
        self.allow_mutation = allow_mutation
        self.favor_best = favor_best
        self.last_fitness = np.zeros(shape=generation_size)
        self.fitness_history = []
        self.__init_first_generation()

    def __init_first_generation(self):
        self.all_specimens: List[Specimen] = \
            np.array([self.specimen_type(**self.init_args)
                     for _ in range(self.generation_size)])

    def simulate(self, n_generations, print_interval=5):
        for i in range(n_generations):
            if (i+1) % print_interval == 0:
                print(f'Starting Generation: {i+1}... ', end="")
            self.simulate_one_generation()
            if (i+1) % print_interval == 0:
                self.fitness_history.append(self.last_fitness.max())
                print(
                    f'Finished! Best fitness: {self.last_fitness.max()}')
        return self.fitness_history

    def simulate_one_generation(self):
        new_specimens = np.empty(self.generation_size, dtype=Specimen)

        # evaluate all current specimen
        fitness = np.array([
            self.all_specimens[i].evaluate() for i in range(self.generation_size)
        ])
        self.last_fitness = fitness
        if np.all(np.isinf(fitness)):
            print("all fitness are -infinity")

        # hold out best
        top_idx = np.argpartition(fitness, -self.top_k)[-self.top_k:]
        new_specimens[:self.top_k] = self.all_specimens[top_idx]

        # create the remaining
        weights = softmax(fitness)
        for i in range(self.top_k, self.generation_size):
            new_specimens[i] = self.__create_new_specimen(weights)

        # update
        self.all_specimens = new_specimens

    def __create_new_specimen(self, weights=None):
        if weights is None:
            weights = np.ones(self.generation_size)
        n_parents = N_PARENTS_IF_CROSSOVER if self.allow_crossover else 1
        parents = np.random.choice(
            self.all_specimens, size=n_parents, replace=False, p=weights)
        new_specimen: Specimen = self.specimen_type.crossover(*parents)
        if self.allow_mutation:
            new_specimen.mutate(**self.mutation_args)
        return new_specimen

    def get_best_specimen(self):
        fitness = np.array([
            self.all_specimens[i].evaluate() for i in range(self.generation_size)
        ])
        return self.all_specimens[fitness.argmax()], fitness.max()


def main():
    # from psqt import pesto_psqt
    # import cProfile
    # import re

    # PSQTCombination.use_target_psqts(pesto_psqt)
    # evolution = Evolution(
    #     PSQTCombination,
    #     generation_size=1_000,
    #     top_k=25,
    #     allow_crossover=True,
    #     mutation_args={'weight_p': 0.03, 'component_p': 0.002})

    # def foo():
    #     for _ in range(50_000):
    #         evolution.all_specimens[0].mutate()
    # cProfile.runctx('foo()', None, locals())
    pass


if __name__ == '__main__':
    main()
