from __future__ import annotations
import argparse
import os
import json
import pulp
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from enum import Enum
from typing import Dict, List, Tuple, Optional


# A basis object contains a list of indices corresponding to the constaints
# of A in the linear program.
Basis = List[int]

class Status(int, Enum):
    CANDIDATE = 0  # Candidate BFS (but no other information)
    OPTIMAL = 1  # Candidate is optimal
    INFEASIBLE = 2  # Problem is infeasible
    UNBOUNDED = 3  # Problem is unbounded

class LinearProgram:
    def __init__(self, A: np.array, b: np.array, c: np.array) -> None:
        """ Linear program of the form

            min c.T x
            s.t. a x = b
                  x >= 0
                  b >= 0 (to simplify finding initial feasible solution)

            Uses dense matrix and vector representations.

            You may NOT modify this method
        """

        self.A = A  # m x n
        self.b = b  # m
        self.c = c  # n
        self.x = np.zeros_like(self.c)  # n

    @classmethod
    def load(cls, f: str) -> LinearProgram:
        """ Loads LP from one of the lp-*.json files
            You may NOT modify this method
        """
        data = json.load(open(f))
        n = data["n"]
        m = len(data["a"])
        A = np.zeros((m, n))
        b = np.zeros((m))
        c = np.zeros((n))

        for i, con in enumerate(data["a"]):
            for j, a_ij in con[0]:
                A[i, j] = a_ij
            b[i] = con[1]

        for j, c_j in data["c"]:
            c[j] = c_j
        return cls(A, b, c)


    def starting_bfs_lp(self) -> Tuple[LinearProgram, Basis]:
        """ Returns a modified LP used for finding a starting BFS with the
            corresponding basis of slack variables
        """
        m, n = self.A.shape

        # Add artificial variables to A
        artificial_matrix = np.eye(m)
        A_modified = np.hstack((self.A, artificial_matrix))
    
        # Modify b (it remains the same in this case)
        b_modified = self.b
    
        # Modify c to minimize the sum of artificial variables
        c_modified = np.hstack((np.zeros(n), np.ones(m)))
    
        # The basis for the modified LP consists of the artificial variables
        basis = list(range(n, n+m))
    
        return LinearProgram(A_modified, b_modified, c_modified), basis


def simplex_step(lp: LinearProgram, basis: Basis) -> Tuple[Status, Optional[np.array(int)]]:
    """ One step of the simplex algorithm
        Returns a tuple containing the status of the solver, and the updated basis
        if status is CANDIDATE, else None.
    """
    # Calculate reduced costs
    c_b = lp.c[basis]
    B = lp.A[:, basis]
    B_inv = np.linalg.inv(B)
    reduced_costs = lp.c - c_b.T @ B_inv @ lp.A

    # Check for optimality
    if np.all(reduced_costs >= 0):
        return Status.OPTIMAL, basis

    # Determine entering variable
    entering_var = np.argmin(reduced_costs)

    # Determine leaving variable
    y = B_inv @ lp.A[:, entering_var]
    ratios = lp.b / y
    if np.all(y <= 0):
        return Status.UNBOUNDED, None
    leaving_var_index = np.argmin(ratios)
    leaving_var = basis[leaving_var_index]

    # Pivot
    basis[leaving_var_index] = entering_var

    return Status.CANDIDATE, basis

def simplex_step_opt(lp: LinearProgram, basis: Basis) -> Tuple[Status, Optional[np.array(int)]]:
    """ One step of the simplex algorithm. Uses the LU factorisation optimisation.
        Returns a tuple containing the status of the solver, and the updated basis
        if status is CANDIDATE, else None.
    """
    # Calculate reduced costs using LU factorization
    c_b = lp.c[basis]
    B = lp.A[:, basis]
    lu, piv = lu_factor(B)
    pi = lu_solve((lu, piv), c_b)
    reduced_costs = lp.c - pi @ lp.A

    # Check for optimality
    if np.all(reduced_costs >= 0):
        return Status.OPTIMAL, basis

    # Determine entering variable
    entering_var = np.argmin(reduced_costs)

    # Determine leaving variable using LU factorization
    y = lu_solve((lu, piv), lp.A[:, entering_var])
    ratios = lp.b / y
    if np.all(y <= 0):
        return Status.UNBOUNDED, None
    leaving_var_index = np.argmin(ratios)
    leaving_var = basis[leaving_var_index]

    # Pivot
    basis[leaving_var_index] = entering_var

    return Status.CANDIDATE, basis
def simplex(lp: LinearProgram, opt: bool) -> Dict:
    """
    Solves a LinearProgram using simplex algorithm

    lp: a LinearProgram
    opt: a flag deciding whether you want to use the LU decomposition optimisation or not

    Returns a dict with the following entries:
        status: Status [always]
        bfs_pivots: number of pivots to find BFS [always]
        opt_pivots: number of pivots from BFS onwards [when UNBOUNDED or OPTIMAL]
        x: numpy array of variable values in same order as LinearProgram [when OPTIMAL]
        basis: variable indices of the current BFS [when OPTIMAL]
        objective: objective value [when OPTIMAL]
    """
    """ Solves a LinearProgram using simplex algorithm """
        # Initializations
    status = Status.CANDIDATE
    bfs_pivots = 0
    opt_pivots = 0
    basis = None  # Initialize with the starting BFS

    while status == Status.CANDIDATE:
        if opt:
            status, new_basis = simplex_step_opt(lp, basis)
        else:
            status, new_basis = simplex_step(lp, basis)
        
        if status == Status.CANDIDATE:
            basis = new_basis
            opt_pivots += 1
        elif status in [Status.OPTIMAL, Status.UNBOUNDED]:
            break

    x = lp.x  # Solution vector
    objective = np.dot(lp.c, x)  # Objective value

    return {
        "status": status,
        "bfs_pivots": bfs_pivots,
        "opt_pivots": opt_pivots,
        "x": x,
        "basis": basis,
        "objective": objective
    }


if __name__ == "__main__":
    """ You are free to modify the code below """
    parser = argparse.ArgumentParser("Simplex Linear Programming Solver")
    parser.add_argument("file", type=str,
                        help="json model file")
    parser.add_argument("--opt", action="store_true",
                        help="use the LU decomposition optimisation")
    args = parser.parse_args()

    lp = LinearProgram.load(args.file)
    sol = simplex(lp, args.opt)

    print(f'Status: {sol["status"]}')
    if sol["status"] == Status.OPTIMAL:
        print(f'Objective: {sol["objective"]}')
        print(f'Variables: {sol["x"]}')
        print(f'Basis: {sol["basis"]}')
    if sol["status"] == Status.INFEASIBLE:
        print(f'Pivots: {sol["bfs_pivots"]}')
    else:
        print(f'Pivots (BFS, OPT): {sol["bfs_pivots"]} {sol["opt_pivots"]}')
