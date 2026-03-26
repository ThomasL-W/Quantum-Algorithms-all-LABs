# Optimization Library

**Authors:** [Student 1], [Student 2]  
**Deadline:** 24/03 at midnight CET

---

## Project Structure

```
optimization_project/
│
├── Problem/
│   ├── __init__.py
│   ├── MaxCutProblem.py      # Max-Cut on weighted/unweighted graphs
│   ├── QuboProblem.py        # QUBO with upper triangular or symmetric matrix
│   ├── IsingProblem.py       # Ising model from graph or (w, J) arrays
│   ├── KnapsackProblem.py    # 0/1 Knapsack with feasibility check
│   ├── TspProblem.py         # Travelling Salesman Problem
│   └── Converter.py          # qubo_to_ising / ising_to_qubo conversions
│
└── Algorithm/
    ├── __init__.py
    └── Classical_Algorithm/
        ├── __init__.py
        ├── Exhaustive_search.py    # Tries all solutions (+ parallel version)
        ├── Random_search.py        # Randomly samples configurations
        ├── Local_search.py         # Greedy descent from random start
        └── Random_local_search.py  # RLS: random restarts + local search (Bonus TODO 23)
```

---

## Usage

```python
from optimization_project import *

# Max-Cut
import networkx as nx
G = nx.Graph()
G.add_edges_from([(0,1),(1,2),(2,3),(3,0)])
problem = MaxCutProblem(G)
best, cost = exhaustive_search(problem, maximize=False)

# Knapsack
knapsack = KnapsackProblem([10, 6, 5], [5, 4, 3], capacity=8)
best, value = random_search(knapsack, n_iterations=500, maximize=True)

# QUBO local search
import numpy as np
Q = np.array([[1, 2], [0, 3]], dtype=float)
qubo = QuboProblem(Q)
best, cost = local_search(qubo, n_iterations=1000)

# Random Local Search (bonus)
best, cost = random_local_search(qubo, n_restarts=10, n_iterations=200)
```
