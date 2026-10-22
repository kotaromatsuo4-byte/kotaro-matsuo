import numpy as np
import cma
from cmaes import CMA


# ============================================
# objective function
# Rastrigin function
# many local minima
# global minimum = 0 at x=[0,0]
# ============================================

def objective(x):

    x = np.asarray(x)

    A = 10

    return (
        A * len(x)
        + np.sum(x**2 - A * np.cos(2*np.pi*x))
    )


# ============================================
# initial condition
# ============================================

dim = 2

x0 = np.array([4.0, 4.0])

sigma0 = 2.0


# ============================================
# 1. Standard CMA-ES
# using cmaes library
# ============================================

print("====================================")
print("Standard CMA-ES")
print("====================================")

optimizer = CMA(
    mean=x0,
    sigma=sigma0
)

max_gen = 100

best_x = None
best_f = 1e10

for generation in range(max_gen):

    solutions = []

    for _ in range(optimizer.population_size):

        x = optimizer.ask()

        value = objective(x)

        solutions.append((x, value))

        if value < best_f:

            best_f = value
            best_x = x.copy()

    optimizer.tell(solutions)

    if optimizer.should_stop():
        break

print()
print("Standard CMA-ES Result")
print("best x =", best_x)
print("best f =", best_f)





