"""
Genetic Algorithms — Target String Problem (Python)

Implements the exact functions from the handout:
- initialize_population(pop_size, length, charset)
- fitness(individual, target)
- select(population, fitnesses)           # tournament selection (k=3)
- crossover(parent1, parent2)
- mutate(individual, mutation_rate, charset)
- run_ga(target, pop_size, generations, mutation_rate)

Adds:
- clean per-generation logging (best/avg fitness, diversity, eval count, etc.)
- matplotlib visualization (convergence curves)
- helpers to run the 5 test cases + experiments
"""

from __future__ import annotations

import random
import string
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict


import matplotlib.pyplot as plt


# ----------------------------
# 1) initialize_population
# ----------------------------
def initialize_population(pop_size, length, charset, rng):
    population = []

    for _ in range(pop_size):
        chars = []
        for _ in range(length):
            chars.append(rng.choice(charset))
        population.append(''.join(chars))

    return population

def fitness(individual, target):
    score = 0

    for a, b in zip(individual, target):
        if a == b:
            score += 1

    return score



def select(population, fitnesses, rng, k=3):
    idxs = rng.sample(range(len(population)), k)

    best_idx = idxs[0]
    best_fit = fitnesses[best_idx]

    for i in idxs:
        if fitnesses[i] > best_fit:
            best_fit = fitnesses[i]
            best_idx = i

    return population[best_idx]

def crossover(parent1, parent2, rng):
    n = len(parent1)
    point = rng.randint(1, n - 1)

    child = ""

    for i in range(n):
        if i < point:
            child += parent1[i]
        else:
            child += parent2[i]

    return child


def mutate(individual, mutation_rate, charset, rng):
    mutated = ""

    for ch in individual:
        if rng.random() < mutation_rate:
            mutated += rng.choice(charset)
        else:
            mutated += ch

    return mutated

# ----------------------------
# GA runner + logging/metrics
# ----------------------------
def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


@dataclass
class GenerationStats:
    gen: int
    best_individual: str
    best_fitness: int
    avg_fitness: float
    unique_ratio: float
    avg_hamming_to_best: float
    evaluations_so_far: int
    elapsed_s: float


def _format_log_line(s: GenerationStats, target_len: int) -> str:
    pct = 100.0 * s.best_fitness / target_len
    return (
        f"gen {s.gen:04d} | "
        f"best {s.best_fitness:2d}/{target_len} ({pct:6.2f}%) | "
        f"avg {s.avg_fitness:6.2f} | "
        f"uniq {s.unique_ratio:5.2f} | "
        f"ham(best) {s.avg_hamming_to_best:5.2f} | "
        f"evals {s.evaluations_so_far:6d} | "
        f"{s.elapsed_s:7.2f}s | "
        f"best='{s.best_individual}'"
    )


# ----------------------------
# 6) run_ga
# ----------------------------


def run_ga(target, pop_size, generations, mutation_rate, charset,
           seed=None, tournament_k=3, elitism=1, log_every=1):

    rng = random.Random(seed)
    start = time.time()

    population = initialize_population(pop_size, len(target), charset, rng)

    best_overall = ""
    best_overall_fit = -1

    stats = []
    evals = 0

    for gen in range(1, generations + 1):

        # ---- evaluate fitness for all individuals ----
        fitnesses = []
        for ind in population:
            fitnesses.append(fitness(ind, target))
        evals += pop_size

        # ---- find best in this generation + average fitness ----
        best_ind = population[0]
        best_fit = fitnesses[0]
        total_fit = 0

        for i in range(pop_size):
            total_fit += fitnesses[i]
            if fitnesses[i] > best_fit:
                best_fit = fitnesses[i]
                best_ind = population[i]

        avg_fit = total_fit / pop_size

        # ---- track overall best ----
        if best_fit > best_overall_fit:
            best_overall_fit = best_fit
            best_overall = best_ind

        # ---- some simple extra stats ----
        unique_ratio = len(set(population)) / pop_size
        elapsed = time.time() - start

        stats.append({
            "gen": gen,
            "best": best_ind,
            "best_fit": best_fit,
            "avg_fit": avg_fit,
            "unique_ratio": unique_ratio,
            "evals": evals,
            "elapsed": elapsed
        })

        # ---- log progress ----
        if log_every > 0 and (gen % log_every == 0 or best_fit == len(target)):
            print(f"gen {gen:3d} | best {best_fit}/{len(target)} | avg {avg_fit:.2f} "
                  f"| uniq {unique_ratio:.2f} | evals {evals} | best='{best_ind}'")

        # ---- stop early if solved ----
        if best_fit == len(target):
            break

        # ---- elitism: keep best N ----
        ranked = []
        for i in range(pop_size):
            ranked.append((fitnesses[i], population[i]))
        ranked.sort(reverse=True)  # sorts by fitness first (tuple)

        next_population = []
        for i in range(elitism):
            next_population.append(ranked[i][1])

        # ---- breed the rest ----
        while len(next_population) < pop_size:
            p1 = select(population, fitnesses, rng, k=tournament_k)
            p2 = select(population, fitnesses, rng, k=tournament_k)

            child = crossover(p1, p2, rng)
            child = mutate(child, mutation_rate, charset, rng)

            next_population.append(child)

        population = next_population

    return best_overall, stats


# ----------------------------
# Plotting helpers
# ----------------------------

def plot_convergence(stats, target, title=""):
    xs   = [s["gen"] for s in stats]
    best = [s["best_fit"] for s in stats]
    avg  = [s["avg_fit"] for s in stats]

    plt.figure()
    plt.plot(xs, best, label="best fitness")
    plt.plot(xs, avg,  label="avg fitness")
    plt.ylim(0, len(target) + 0.5)
    plt.xlabel("generation")
    plt.ylabel("fitness (matches)")
    plt.title(title or f"GA convergence to '{target}'")
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt

def plot_compare(stats_a, label_a, stats_b, label_b):
    xa = [s["gen"] for s in stats_a]
    ya = [s["best_fit"] for s in stats_a]

    xb = [s["gen"] for s in stats_b]
    yb = [s["best_fit"] for s in stats_b]

    plt.figure()
    plt.plot(xa, ya, label=label_a)
    plt.plot(xb, yb, label=label_b)
    plt.xlabel("generation")
    plt.ylabel("best fitness")
    plt.title("Best fitness comparison")
    plt.legend()
    plt.show()


# ----------------------------
# Test cases (from the sheet)
# ----------------------------
CHARSET_UPPER = string.ascii_uppercase
CHARSET_UPPER_SPACE = string.ascii_uppercase + " "

TEST_CASES: Dict[str, Tuple[str, str]] = {
    "01": ("HI", CHARSET_UPPER),
    "02": ("EVOLUTION", CHARSET_UPPER),
    "03": ("NATURE INSPIRED", CHARSET_UPPER_SPACE),
    "04": ("AAAAAAAAAA", CHARSET_UPPER),        # (all same chars in TARGET)
    "05": ("GA IS COOL", CHARSET_UPPER_SPACE),
}


def run_case(
    case_id: str,
    pop_size: int = 100,
    generations: int = 200,
    mutation_rate: float = 0.01,
    seed: Optional[int] = 0,
    log_every: int = 1,
) -> Tuple[str, List[GenerationStats]]:
    target, charset = TEST_CASES[case_id]
    print(f"\n--- Case {case_id}: target='{target}' (len={len(target)}) ---")
    best, stats = run_ga(
        target=target,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        charset=charset,
        seed=seed,
        elitism=1,
        tournament_k=3,
        log_every=log_every,
    )
    print(f"Best found: '{best}' (fitness {fitness(best, target)}/{len(target)})")
    plot_convergence(stats, target, title=f"Case {case_id}: '{target}'")
    return best, stats


if __name__ == "__main__":
    # Suggested starting parameters from the handout:
    # target="EVOLUTION", pop_size=100, mutation_rate=0.01, generations=200, charset=uppercase (+ space)
    # run_case("05", pop_size=100, generations=200, mutation_rate=0.01, seed=0, log_every=1)

    # Run the other targets:
    # run_case("01", seed=0)
    # run_case("03", seed=0)
    # run_case("04", seed=0)
    # run_case("05", seed=0)

    # Compare convergence for 02 vs 03 (best fitness curves):
    # _, s02 = run_case("02", seed=0, log_every=0)
    # _, s03 = run_case("03", seed=0, log_every=0)
    # plot_compare(s02, "02: EVOLUTION", s03, "03: NATURE INSPIRED")

    # Experiments:
    # 1) mutation_rate = 0 (often stagnates once population loses needed alleles)
    # run_case("02", mutation_rate=0.0, seed=0)

    # 2) mutation_rate = 0.5 (very noisy; selection struggles to keep improvements)
    # run_case("02", mutation_rate=0.5, seed=0)

    # 3) pop_size = 5 (low diversity; more premature convergence)
    # run_case("02", pop_size=5, seed=0)

    # 4) run same string 3 times (different seeds)
    for sd in [1, 2, 3]:
        run_case("02", seed=sd, log_every=0)