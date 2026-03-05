#!/usr/bin/env python3

from heapq import heapify, heappop
from operator import attrgetter
from random import choice, randint, random
from dataclasses import dataclass


TARGETS = ["HI", "EVOLUTION", "NATURE INSPIRED", "AAAAAAAAAA", "GA IS COOL"]
TARGET = TARGETS[0]
LENGTH = len(TARGET)
POP_SIZE = 100
MUTATION_RATE = 0.01
GENERATIONS = 200
CHARSET = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'


@dataclass
class Individual:
    state: str

    @property
    def fitness(self, target: str) -> list[int]:
        perIndexFitness = []
        for i in range(len(self.state)):
            perIndexFitness.append(ord(self.state[i]) - ord(target[i]))
        return perIndexFitness

    @property
    def totalFitness(self, target: str) -> int:
        return self.fitness.sum


def initialize_population(pop_size, length, charset):
    population = []
    s = ''
    for i in range(pop_size):
        for j in range(length):
            s.append(choice(charset))
        population.append(Individual(s))
        s = ''
    return population


def select(population: list[Individual]):
    SELECTION_LENGTH = 3
    
    heap = population[:]
    heapify(heap, key=lambda t: -t.priority) # per me bo max heap
    
    output = []
    for i in range(SELECTION_LENGTH):
        output.append(heappop(heap))

    return output


def crossover(parent1, parent2):
    s = ''
    for i in range(LENGTH):
        if parent1.fitness[i] <= parent2.fitness[i]:
            s.append(parent1[i])
        else:
            s.append(parent2[i])
    return s


def mutate(individual, mutation_rate, charset):
    
    rnd = random()
    if rnd > mutation_rate:
        return individual

    idx = randint(0, LENGTH-1)

    mutation = choice(charset)
    individual[idx] = mutation

    return individual


def run_ga(target, pop_size, generations: list[Individual], mutation_rate):
    generation = initialize_population(POP_SIZE, LENGTH, CHARSET)
    for i in generations:
        select(generation)
        for 