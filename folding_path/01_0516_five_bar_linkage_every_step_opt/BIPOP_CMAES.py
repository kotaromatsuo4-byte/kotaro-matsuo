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
# 2. BIPOP-CMA-ES
# using pycma library
# ============================================

print()
print("====================================")
print("BIPOP-CMA-ES")
print("====================================")

# ============================================
# BIPOP-CMA-ES
# ============================================

result = cma.fmin(
    objective,
    x0,
    sigma0,

    options={
        "verb_disp": 0,
        "verbose": -9,
        "maxiter": 100,
    },

    restarts=9,
    bipop=True
)


# ============================================
# result
# ============================================

best_x = result[0]
best_f = result[1]

print()
print("BIPOP-CMA-ES Result")
print("best x =", best_x)
print("best f =", best_f)



