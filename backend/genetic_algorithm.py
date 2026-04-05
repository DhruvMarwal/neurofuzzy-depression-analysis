"""
SECTION 2: GENETIC ALGORITHM (GA) WEIGHT OPTIMIZER
=====================================================
Uses DEAP (Distributed Evolutionary Algorithms in Python) to evolve
optimal weights for each of the 9 PHQ-9 questions.

Why GA?
  Standard PHQ-9 treats all 9 questions equally (weight = 1).
  But clinically, question 9 (suicidal ideation) is far more critical
  than question 3 (sleep issues). GA discovers near-optimal weights
  by evolving a population of candidate weight vectors and selecting
  those that best separate severity classes on labelled data.

GA Parameters:
  Population    : 100 individuals
  Generations   : 50
  Crossover     : Uniform (cx_pb = 0.7)
  Mutation      : Gaussian noise (mut_pb = 0.2, sigma = 0.1)
  Selection     : Tournament (k=3)
  Fitness       : Minimise classification error on training data
"""

import random
import numpy as np
from deap import base, creator, tools, algorithms
from phq9_scorer import FuzzyPHQ9Scorer


# ── Synthetic Training Dataset ─────────────────────────────────────────────────
# Format: (responses[9], true_label_index)
# Labels: 0=Minimal, 1=Mild, 2=Moderate, 3=ModeratelySevere, 4=Severe
TRAINING_DATA = [
    # Minimal
    ([0, 0, 0, 0, 0, 0, 0, 0, 0], 0),
    ([0, 1, 0, 0, 0, 0, 0, 0, 0], 0),
    ([1, 0, 1, 0, 0, 0, 0, 0, 0], 0),
    ([0, 0, 0, 1, 1, 0, 0, 0, 0], 0),
    # Mild
    ([1, 1, 1, 1, 1, 0, 0, 0, 0], 1),
    ([2, 1, 1, 0, 1, 1, 0, 0, 0], 1),
    ([1, 2, 1, 1, 0, 1, 0, 1, 0], 1),
    ([0, 1, 2, 1, 1, 1, 1, 0, 0], 1),
    # Moderate
    ([2, 2, 1, 1, 2, 1, 1, 0, 1], 2),
    ([2, 1, 2, 2, 1, 2, 1, 0, 1], 2),
    ([1, 2, 2, 2, 2, 1, 1, 1, 0], 2),
    ([2, 2, 2, 1, 1, 1, 1, 1, 1], 2),
    # Moderately Severe
    ([3, 3, 2, 2, 2, 2, 1, 1, 2], 3),
    ([2, 3, 3, 2, 2, 2, 2, 1, 1], 3),
    ([3, 2, 2, 3, 2, 2, 2, 1, 2], 3),
    ([2, 2, 3, 3, 2, 2, 2, 2, 1], 3),
    # Severe
    ([3, 3, 3, 3, 3, 3, 3, 3, 3], 4),
    ([3, 3, 3, 3, 3, 3, 3, 3, 2], 4),
    ([3, 3, 3, 3, 2, 3, 3, 3, 3], 4),
    ([3, 3, 3, 2, 3, 3, 3, 3, 3], 4),
]

LABEL_THRESHOLDS = [4, 9, 14, 19, 27]   # raw score upper bounds per class


def label_from_score(score: float) -> int:
    """Convert weighted score to label index."""
    for i, thresh in enumerate(LABEL_THRESHOLDS):
        if score <= thresh:
            return i
    return 4


def evaluate(individual) -> tuple:
    """
    Fitness function: proportion of training samples misclassified.
    Lower is better (we minimise).
    """
    weights = list(individual)
    scorer  = FuzzyPHQ9Scorer(weights=weights)
    errors  = 0

    for responses, true_label in TRAINING_DATA:
        result         = scorer.score(responses)
        predicted_label = label_from_score(result["weighted_score"])
        if predicted_label != true_label:
            errors += 1

    # Penalise extreme weights (regularisation)
    penalty = 0.01 * sum((w - 1.0) ** 2 for w in weights)
    fitness  = errors / len(TRAINING_DATA) + penalty
    return (fitness,)


def run_ga(
    n_pop: int = 100,
    n_gen: int = 50,
    cx_pb: float = 0.7,
    mut_pb: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run the Genetic Algorithm to find optimal PHQ-9 weights.

    Returns:
        dict with 'best_weights', 'best_fitness', 'logbook'
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── DEAP setup ─────────────────────────────────────────────────────────
    # Minimisation problem
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Each gene = weight for one PHQ-9 question, initialised in [0.5, 2.0]
    toolbox.register("attr_weight", random.uniform, 0.5, 2.0)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_weight,
        n=9,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",   evaluate)
    toolbox.register("mate",       tools.cxUniform, indpb=0.5)
    toolbox.register(
        "mutate",
        tools.mutGaussian,
        mu=0,
        sigma=0.1,
        indpb=0.3,
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    # ── Evolution ──────────────────────────────────────────────────────────
    population = toolbox.population(n=n_pop)
    hof        = tools.HallOfFame(1)               # Keep best individual
    stats      = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb   = cx_pb,
        mutpb  = mut_pb,
        ngen   = n_gen,
        stats  = stats,
        halloffame = hof,
        verbose = verbose,
    )

    # Clip weights to [0.1, 3.0] for safety
    best_weights = [float(np.clip(w, 0.1, 3.0)) for w in hof[0]]
    best_fitness = float(hof[0].fitness.values[0])

    if verbose:
        print("\n══════════════════════════════════════════")
        print(f"  GA Complete | Best fitness: {best_fitness:.4f}")
        print(f"  Optimal weights: {[round(w, 3) for w in best_weights]}")
        print("══════════════════════════════════════════")

    return {
        "best_weights": best_weights,
        "best_fitness": best_fitness,
        "logbook":      logbook,
    }


def get_default_weights() -> list:
    """
    Return pre-computed GA weights (saved so API can skip re-running GA).
    These are example values; replace with actual GA output after training.
    
    Clinical note:
      Q9 (suicidal ideation) gets highest weight.
      Q3 (sleep problems) and Q4 (fatigue) get lower weights.
    """
    return [1.0, 1.1, 0.9, 0.85, 0.95, 1.05, 1.0, 1.2, 1.5]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running Genetic Algorithm to optimise PHQ-9 weights...\n")
    result = run_ga(n_pop=100, n_gen=50, verbose=True)

    print("\nValidating best weights on training data:")
    scorer = FuzzyPHQ9Scorer(weights=result["best_weights"])
    correct = 0
    for responses, true_label in TRAINING_DATA:
        r       = scorer.score(responses)
        pred    = label_from_score(r["weighted_score"])
        status  = "✓" if pred == true_label else "✗"
        correct += (pred == true_label)
        print(f"  {status} True={true_label}  Pred={pred}  "
              f"Score={r['weighted_score']:.2f}  {r['severity_label']}")

    acc = correct / len(TRAINING_DATA) * 100
    print(f"\nTraining Accuracy: {acc:.1f}%")