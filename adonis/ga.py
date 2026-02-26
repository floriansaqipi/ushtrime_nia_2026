import random
import string
import matplotlib.pyplot as plt

def initialize_population(pop_size, length, charset):
    return ["".join(random.choice(charset) for _ in range(length)) for _ in range(pop_size)]


def fitness(individual, target):
    score = 0
    for i in range(len(target)):
        if individual[i] == target[i]:
            score += 1
    return score


def select(population, fitnesses):
    indices = random.sample(range(len(population)), 3)
    best_index = max(indices, key=lambda i: fitnesses[i])
    return population[best_index]


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child


def mutate(individual, mutation_rate, charset):
    res = list(individual)
    for i in range(len(res)):
        if random.random() < mutation_rate:
            res[i] = random.choice(charset)
    return "".join(res)

def run_ga(target, pop_size=100, generations=300, mutation_rate=0.01, charset=None):
    if charset is None:
        charset = string.ascii_uppercase + " "

    population = initialize_population(pop_size, len(target), charset)
    fitness_history = []

    print(f"\n--- EVOLVING TOWARD: '{target}' ---")

    for gen in range(generations):
        fitnesses = [fitness(ind, target) for ind in population]
        max_fit = max(fitnesses)
        best_ind = population[fitnesses.index(max_fit)]
        fitness_history.append(max_fit)

        if gen % 20 == 0 or max_fit == len(target):
            print(f"Gen {gen:3}: {best_ind} (Fitness: {max_fit})")

        if max_fit == len(target):
            break

        new_population = []
        while len(new_population) < pop_size:
            p1 = select(population, fitnesses)
            p2 = select(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate, charset)
            new_population.append(child)
        population = new_population

    return fitness_history

if __name__ == "__main__":
    test_cases = [
        {"id": "01", "target": "HI", "gens": 50},
        {"id": "02", "target": "EVOLUTION", "gens": 200},
        {"id": "03", "target": "NATURE INSPIRED", "gens": 500},
        {"id": "04", "target": "AAAAAAAAAA", "gens": 100},
        {"id": "05", "target": "GA IS COOL", "gens": 200}
    ]

    plt.figure(figsize=(10, 6))

    for case in test_cases:
        history = run_ga(case["target"], generations=case["gens"])
        plt.plot(history, label=f"Case {case['id']}: {case['target']}")

    plt.title("Convergence Comparison for All Lab Cases")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Matching Characters)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()