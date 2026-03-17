import math
import numpy as np
from sympy import *

# Rekursiver Aufruf von solveset über alle AND- und OR-Relationen eines gegebenen Terms. Liefert eine Liste von Intervallen über q, die den gegebenen Term erfüllen.
# https://docs.sympy.org/latest/modules/solvers/solveset.html: solveset returns a Set representing all of the solutions of a univariate equation.
def solveset_recursive(q_condition, q, q_interval):
    if q_condition == S.true: # Bedingung immer erfüllbar
        return q_interval
    elif q_condition == S.false: # Bedingung niemals erfüllbar
        return S.EmptySet
    elif isinstance(q_condition, And): # AND: intersect über alle Argumente
        q_set = q_interval
        for arg in q_condition.args:
            q_subset = solveset_recursive(arg, q, q_interval)
            q_set = q_set.intersect(q_subset)
        return q_set
    elif isinstance(q_condition, Or): # OR: union über alle Argumente
        q_sol = S.EmptySet
        for arg in q_condition.args:
            q_subset = solveset_recursive(arg, q, q_interval)
            q_sol = q_sol.union(q_subset)
        q_set = q_sol.intersect(q_interval)
        return q_set
    else: # Lösung durch solveset
        q_sol = solveset(q_condition, q, domain=S.Reals)
        q_set = q_sol.intersect(q_interval)
        return q_set

# True, falls eine gegebene Bedingung für einen konkreten Wahrscheinlichkeitswert q erfüllt wird, sonst False
def probability_val_check(se_condition, q, grid_q):
    grid_cond = simplify(se_condition.subs(q, grid_q))
    if grid_cond == S.true:
        return True
    se_rels = list(grid_cond.args) if isinstance(grid_cond, And) else [grid_cond]
    if reduce_inequalities(se_rels) == S.true:
        return True
    return False

# Sucht eine Belegung einer gegebenen se_condition durch Einsetzen von Wahrscheinlichkeitswerten aus einem festen grid
def probability_grid_check(se_condition, q, grid_step = 0.01):
    prob_grid = np.array([[0.5+delta, 0.5-delta] for delta in np.arange(0, 0.5+grid_step, grid_step)]).flatten()[1:] # prob_grid = [0.5, 0.51, 0.49, 0.52, 0.48, ..., 1.0, 0.0]
    for grid_q in prob_grid:
        if probability_val_check(se_condition, q, grid_q) == S.true:
            return True, grid_q
    return False, math.nan

# Vergleich unterschiedlicher solver für eine gegebenen Ungleichung
def inequalities_solver_test(q_condition, q, q_interval):

    assert(isinstance(q_condition, And))
    q_solutions = {}

    # Allgemeine Lösung einer gegebenen Bedingung für q durch rekursives solveset
    # https://docs.sympy.org/latest/modules/solvers/solveset.html: solveset returns a Set representing all of the solutions of a univariate equation.
    q_solutions["solveset_recursive"] = solveset_recursive(q_condition, q, q_interval)

    # Fallback, falls solveset fehlschlägt: Lösung durch solve. solve() garantiert aber nicht, dass eine oder alle möglichen Lösungen gefunden werden, siehe https://docs.sympy.org/latest/modules/solvers/solveset.html: 
    # There are cases where solve returns an empty list. This might mean that there are no solutions or no solution could be found given its currently supported features. 
    # In other cases, there is no way (given the output interface of solve) to communicate whether or not:
    # * all possible solutions to the system were found,
    # * or that there are provably no solutions,
    # * or that the real set of solutions are finite or infinite,
    # * etc.
    # * solve may return a few solutions when more solutions (potentially infinitely many) also exist.
    q_solutions["solve"] = solve(list(q_condition.args), q)

    # Numerische Näherung durch Einsetzen von Wahrscheinlichkeitswerten aus einem festen grid
    q_solutions["probability_grid_check"] = probability_grid_check(q_condition, q)
    q_solutions["probability_val_check(q=0.37)"] = probability_val_check(q_condition, q, 0.37)
    q_solutions["probability_val_check(q=0.5)"] = probability_val_check(q_condition, q, 0.5)

    print(f"q_condition = {q_condition}")
    for name, q_sol in q_solutions.items():
        print(f"{name}: {q_sol}")

if __name__ == "__main__":

    q = symbols("q", real = True)
    q_condition = (0 < q) & (q < 1) & (((-1.7147482907102 <= q) & (q <= -0.999999999999995)) | ((-0.0251216147175559 <= q) & (q <= 0.664211911758481)) | ((q <= -0.974342006330726) & (-0.999999999999772 < q)) | ((-0.973646744536023 < q) & (q < -0.973646592749041))) & (((-1.42389487053596 <= q) & (q <= -0.999999999999989)) | ((-0.0245356944581915 <= q) & (q <= 0.372772929366745)) | ((q <= -0.974342364372601) & (-0.999999999999772 < q)) | ((-0.973646744536023 < q) & (q < -0.973646592749041)))
    inequalities_solver_test(q_condition, q, Interval.open(0, 1))
