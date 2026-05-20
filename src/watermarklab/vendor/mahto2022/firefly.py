from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class FireflyResult:
    best_alpha: float
    best_fitness: float
    history: list[dict]


def firefly_optimize(
    objective: Callable[[float], float],
    lower: float = 0.02,
    upper: float = 0.20,
    n_fireflies: int = 6,
    iterations: int = 4,
    beta0: float = 1.0,
    gamma: float = 1.0,
    randomization: float = 0.02,
    seed: int = 123,
) -> FireflyResult:
    """Small Firefly optimizer for embedding strength alpha.

    The objective is minimized. This matches the paper's idea of minimizing a
    fitness penalty from PSNR/NC/BER terms, while keeping parameters explicit.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(lower, upper, size=n_fireflies)
    fitness = np.asarray([objective(float(v)) for v in x], dtype=np.float64)
    history = []

    for it in range(iterations):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] < fitness[i]:
                    r = abs(x[i] - x[j]) / max(upper - lower, 1e-12)
                    beta = beta0 * np.exp(-gamma * r * r)
                    step = beta * (x[j] - x[i]) + randomization * rng.normal()
                    candidate = float(np.clip(x[i] + step, lower, upper))
                    candidate_fitness = objective(candidate)
                    if candidate_fitness < fitness[i]:
                        x[i] = candidate
                        fitness[i] = candidate_fitness
        best_idx = int(np.argmin(fitness))
        history.append({"iteration": it + 1, "best_alpha": float(x[best_idx]), "best_fitness": float(fitness[best_idx])})

    best_idx = int(np.argmin(fitness))
    return FireflyResult(best_alpha=float(x[best_idx]), best_fitness=float(fitness[best_idx]), history=history)
