# Lagrangian Heuristic for the Uncapacitated Facility Location Problem (UFL)

This repository contains a Python implementation of a **Lagrangian heuristic** for the **Uncapacitated Facility Location Problem (UFL)**.
The project focuses on computing strong **lower and upper bounds** using Lagrangian relaxation and subgradient optimization.

---

## 📌 Problem Overview

The Uncapacitated Facility Location Problem (UFL) determines:
- which facilities to open, and
- how to assign each market to exactly one open facility,

while minimizing total fixed opening costs and transportation costs.

---

## 🧠 Solution Approach

The solution is based on a **Lagrangian relaxation** of the assignment constraints:

- Relaxed constraints are penalized using Lagrange multipliers
- A **lower bound (LB)** is obtained from the relaxed problem
- A **feasible solution** is constructed to compute an **upper bound (UB)**
- Lagrange multipliers are updated using a **subgradient method**
- The algorithm iterates until the optimality gap is sufficiently small

A detailed mathematical formulation and algorithmic description can be found in the project report.

---


## 📂 Repository Structure


```
ufl-lagrangian-heuristic/
│
├── src/
│ └── UFL.py
│
├── Instances/
│ └── MO1 – MO5
│
├── outputs/
│ └── progression_of_bounds_over_iterations.png
│
├── UFL_Report.pdf
│
└── README.md
```

