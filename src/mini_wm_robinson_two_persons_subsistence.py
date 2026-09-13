"""
Mini-WM: Ein minimalistisches Wirtschaftsmodell,
Teil 2: Robinson-Crusoe-Wirtschaft mit 2 Personen: Robinson und Freitag.
Erläuterungen siehe MiniWM_Teil02_RobinsonFreitag.md.
"""
from enum import Enum
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
import pathlib
import scipy # pip install scipy
import scipy.optimize
from sympy import * # pip install sympy

# assert and raise exception
def assert_or_die(assert_cond, msg):
    # assert assert_cond, msg
    if not assert_cond:
        print(f"## ERROR {__file__}: {assert_cond} failed, {msg}")
        raise(Exception(f"{assert_cond} failed, {msg}"))

# Shortcut zur Formatierung aller Elemente einer Liste
def fmt(list, fmt_str = "{:5.2f}", sep = ",", pre = "[", post="]"):
    return pre + sep.join([fmt_str.format(_elem) for _elem in list]) + post

# Speichert den aktuellen plot als png file
def save_plot_png(pngfile):
    if pngfile != "":
        pathlib.Path(pngfile).unlink(missing_ok=True)
        plt.savefig(pngfile)
        print(f"  Plot gespeichert in {pngfile}")

# Parameter grid_x_steps bzw. x_steps in rcw_02_grid_search_iter bzw. rcw_02_grid_search:
# step_type=DELTA_X: x-Werte werden um x_steps[i] inkrementiert, step_type=NUM_STEPS: x-Intervall hat x_steps[i] Werte
class GridStepType(Enum):
    DELTA_X = 1
    NUM_STEPS = 2

# Lösung von rcw_02_subsistence mit grid_search (Maximierung von U_fun(x) mittels Suche über alle Kombinationen in einem Gitter mit gegebener Schrittweite)
def rcw_02_grid_search(U_fun, x_start, x_stop, x_steps, step_type:GridStepType, U_min, constraints_fun = None, verbose=0):
    best_x = np.array(x_start, dtype=np.float32)
    best_y = U_min
    if step_type == GridStepType.DELTA_X:
        x_ranges = [np.arange(start, stop, step) for start, stop, step in zip(x_start, x_stop, x_steps)]
    else:
        x_ranges = [np.linspace(start, stop, step) for start, stop, step in zip(x_start, x_stop, x_steps)]
    for x in itertools.product(*x_ranges):
        if constraints_fun is not None and not constraints_fun(x):
            continue
        y = U_fun(x)
        if y is not nan and y > best_y:
            best_x = np.array(x, dtype=np.float32)
            best_y = y
            if verbose > 1:
                print(f"    grid_search: x={x}, y={y}")
    return best_x, best_y

# Lösung von rcw_02_subsistence für Fall 1 (Autarkie) mit Iteration über grid_search
# Suche max(U_fun(x)) mit x = [L_fisch, L_nuss]
def rcw_02_grid_search_iter(U_fun, grid_x_start, grid_x_stop, grid_x_steps, step_type:GridStepType=GridStepType.DELTA_X, num_iter=3, constraints_fun = None, verbose=0):
    assert_or_die(len(grid_x_start) == len(grid_x_stop) == len(grid_x_steps), "grid_x_start, grid_x_stop und grid_x_steps müssen die gleiche Länge haben")
    sol_grid_x = grid_x_start
    sol_grid_U = 0
    for grid_iter in range(num_iter):
        grid_x, grid_U = rcw_02_grid_search(U_fun, x_start=grid_x_start, x_stop=grid_x_stop, x_steps=grid_x_steps, step_type=step_type, U_min = sol_grid_U, constraints_fun = constraints_fun, verbose = verbose)
        if grid_U > sol_grid_U:
            sol_grid_x = grid_x
            sol_grid_U = grid_U
            if verbose > 0:
                print(f"    grid_search: sol_grid_x={sol_grid_x}, sol_grid_U={sol_grid_U:.2f}")
            for i in range(len(grid_x_start)):
                if step_type == GridStepType.DELTA_X:
                    dx = grid_x_steps[i]
                    grid_x_steps[i] *= 0.5
                else:
                    dx = (grid_x_stop[i] - grid_x_start[i]) / (grid_x_steps[i] + 1)
                grid_x_start[i] = max(sol_grid_x[i] - dx, grid_x_start[i])
                grid_x_stop[i]  = min(sol_grid_x[i] + dx, grid_x_stop[i])
        else:
            break
    return sol_grid_x, sol_grid_U

# Robinson-Crusoe-Wirtschaft mit 2 Personen (Robinson und Freitag) mit reiner Subsistenzwirtschaft.
# Die Produktion besteht aus Fischen und Kokosnüssen und wird vollständig zur Subsistenz konsumiert.
def rcw_02_subsistence_fish_nuts():
    
    # Beispielparameter
    alpha_fisch = 0.5 # Elastizität von Arbeit in der Fischproduktion
    alpha_nuss = 0.4  # Elastizität von Arbeit in der Kokosnussproduktion
    A1_fisch = 20 / (12**alpha_fisch) # Robinsons Produktivität bei der Fischproduktion Y1_fisch = A1_fisch * L1_fisch^alpha_fisch, Robinson fängt 20 Fische in 12 Stunden
    A2_fisch = 10 / (12**alpha_fisch) # Freitags Produktivität bei der Fischproduktion Y2_fisch = A2_fisch * L2_fisch^alpha_fisch, Freitag fängt 10 Fische in 12 Stunden
    A1_nuss = 15 / (12**alpha_nuss)   # Robinsons Produktivität bei der Kokosnussproduktion Y1_nuss = A1_nuss * L1_nuss^alpha_nuss, Robinson erntet 15 Kokosnüsse in 12 Stunden
    A2_nuss = 30 / (12**alpha_nuss)   # Freitags Produktivität bei der Kokosnussproduktion Y2_nuss = A2_nuss * L2_nuss^alpha_nuss, Freitag erntet 30 Kokosnüsse in 12 Stunden

    # Shortcuts für alle Elemente in _list >= 0
    ge0 = lambda _list : all(_elem >= 0 for _elem in _list) # true, wenn alle Elemente in _list >= 0 sind

    # Cobb-Douglas-Produktionsfunktionen für Robinson (Y1_fisch, Y1_nuss) und Freitag (Y2_fisch, Y2_nuss)
    Y1_fisch = lambda L1_fisch: A1_fisch * L1_fisch ** alpha_fisch if ge0([L1_fisch]) else np.nan
    Y2_fisch = lambda L2_fisch: A2_fisch * L2_fisch ** alpha_fisch if ge0([L2_fisch]) else np.nan
    Y_fisch = lambda L1_fisch, L2_fisch: Y1_fisch(L1_fisch) + Y2_fisch(L2_fisch)
    Y1_nuss = lambda L1_nuss: A1_nuss * L1_nuss ** alpha_nuss if ge0([L1_nuss]) else np.nan
    Y2_nuss = lambda L2_nuss: A2_nuss * L2_nuss ** alpha_nuss if ge0([L2_nuss]) else np.nan
    Y_nuss = lambda L1_nuss, L2_nuss: Y1_nuss(L1_nuss) + Y2_nuss(L2_nuss)

    # Nutzenfunktionen für Robinson (U1) und Freitag (U2):
    # U_i = A_i_subsistenz * L_i_subsistenz^α_i_subsistenz * C_i_fisch_subsistenz^β_i_subsistenz * C_i_nuss_subsistenz^γ_i_subsistenz
    alpha_subsistenz = 2 # Subsistenz mit höherer Elastizität, hier vereinfacht α_i_subsistenz = β_i_subsistenz = γ_i_subsistenz = 2
    A1_subsistenz = 1 / ((10**alpha_subsistenz) * ((20)**alpha_subsistenz) * (10**alpha_subsistenz)) # Normierung: U1 = 1 bei L_subsistenz = 10 Stunden sowie 20 konsumierten Fischen und 10 konsumierten Kokosnüssen
    A2_subsistenz = 1 / ((8**alpha_subsistenz) * ((16)**alpha_subsistenz) * (15**alpha_subsistenz))  # Normierung: U2 = 1 bei L_subsistenz =  8 Stunden sowie 16 konsumierten Fischen und 15 konsumierten Kokosnüssen
    U1 = lambda L1_subsistenz, C1_fisch_subsistenz, C1_nuss_subsistenz: A1_subsistenz * (L1_subsistenz**alpha_subsistenz) * (C1_fisch_subsistenz**alpha_subsistenz) * (C1_nuss_subsistenz**alpha_subsistenz) if ge0([L1_subsistenz, C1_fisch_subsistenz, C1_nuss_subsistenz]) else np.nan
    U2 = lambda L2_subsistenz, C2_fisch_subsistenz, C2_nuss_subsistenz: A2_subsistenz * (L2_subsistenz**alpha_subsistenz) * (C2_fisch_subsistenz**alpha_subsistenz) * (C2_nuss_subsistenz**alpha_subsistenz) if ge0([L2_subsistenz, C2_fisch_subsistenz, C2_nuss_subsistenz]) else np.nan

    # Zeitrestriktion auf 24 Stunden: L1_subsistenz = L_sum - L1_fisch - L1_nuss, L2_subsistenz = L_sum - L2_fisch - L2_nuss
    L_sum = 24

    ##
    ## Fall 1: Autarkie, Robinson und Freitag produzieren und konsumieren jeweils nur für sich selbst. 
    ## Jeder maximiert seine eigene Nutzenfunktion U_i unabhängig vom anderen.
    ##

    # Fall 1, Autarkie, 2 Unbekannte: x[0] = L1_fisch, x[1] = L1_nuss
    U1_fun = lambda x: U1(L1_subsistenz = L_sum - x[0] - x[1], C1_fisch_subsistenz = Y1_fisch(x[0]), C1_nuss_subsistenz = Y1_nuss(x[1]))
    U2_fun = lambda x: U2(L2_subsistenz = L_sum - x[0] - x[1], C2_fisch_subsistenz = Y2_fisch(x[0]), C2_nuss_subsistenz = Y2_nuss(x[1]))
    # Lösung per grid_search (Plausibilitätscheck und Startwerte für die Lagrange-Lösung)
    print(f"Reine Subsistenzwirtschaft, 2 Personen, Güter sind Fische und Kokosnüsse, Autarkie:")
    sol1_grid_x, sol1_grid_U = rcw_02_grid_search_iter(U_fun = U1_fun, grid_x_start = [0, 0], grid_x_stop = [24, 24], grid_x_steps = [0.1, 0.1], step_type=GridStepType.DELTA_X, num_iter = 3, constraints_fun = None)
    sol2_grid_x, sol2_grid_U = rcw_02_grid_search_iter(U_fun = U2_fun, grid_x_start = [0, 0], grid_x_stop = [24, 24], grid_x_steps = [0.1, 0.1], step_type=GridStepType.DELTA_X, num_iter = 3, constraints_fun = None)
    print(f"  grid-search: L1_fisch={sol1_grid_x[0]:4.1f}, L1_nuss={sol1_grid_x[1]:4.1f}, L1_subsistenz={L_sum-sol1_grid_x[0]-sol1_grid_x[1]:4.1f}, Y1_fisch=C1_fisch={Y1_fisch(sol1_grid_x[0]):5.2f}, Y1_nuss=C1_nuss={Y1_nuss(sol1_grid_x[1]):5.2f}, U1={sol1_grid_U:.2f}")
    print(f"  grid-search: L2_fisch={sol2_grid_x[0]:4.1f}, L2_nuss={sol2_grid_x[1]:4.1f}, L2_subsistenz={L_sum-sol2_grid_x[0]-sol2_grid_x[1]:4.1f}, Y2_fisch=C2_fisch={Y2_fisch(sol2_grid_x[0]):5.2f}, Y2_nuss=C2_nuss={Y2_nuss(sol2_grid_x[1]):5.2f}, U2={sol2_grid_U:.2f}")
    # Lösung dU/dx = 0 mittels Lagrange
    x11, x12, x13, lambda1 = symbols("L1_fisch L1_nuss L1_subsistenz lambda1", real=True, positive=True)
    x21, x22, x23, lambda2 = symbols("L2_fisch L2_nuss L2_subsistenz lambda2", real=True, positive=True)
    # Lagrangefunktion: L = U(x1, x2, x3) + λ*(24 - x1 - x2 - x3)
    U1_x1 = lambda x11, x12, x13: U1(L1_subsistenz = x13, C1_fisch_subsistenz = Y1_fisch(x11), C1_nuss_subsistenz = Y1_nuss(x12))
    U2_x2 = lambda x21, x22, x23: U2(L2_subsistenz = x23, C2_fisch_subsistenz = Y2_fisch(x21), C2_nuss_subsistenz = Y2_nuss(x22))
    L1 = U1_x1(x11, x12, x13) + lambda1 * (L_sum - x11 - x12 - x13)
    L2 = U2_x2(x21, x22, x23) + lambda2 * (L_sum - x21 - x22 - x23)
    # First-Order-Conditions (FOCs): ∂L/∂x1 = 0, ∂L/∂x2 = 0, ∂L/∂x3 = 0, ∂L/∂λ = 0
    L1_vars = [x11, x12, x13, lambda1]
    L1_start = [sol1_grid_x[0], sol1_grid_x[1], L_sum-sol1_grid_x[0]-sol1_grid_x[1], 1]
    L1_focs = [simplify(diff(L1, var)) for var in L1_vars]
    sol1_equilibriums = nsolve(L1_focs, L1_vars, L1_start)
    X1_autarkie = sol1_equilibriums[0:3]
    U1_autarkie = U1(L1_subsistenz = X1_autarkie[2], C1_fisch_subsistenz = Y1_fisch(X1_autarkie[0]), C1_nuss_subsistenz = Y1_nuss(X1_autarkie[1]))
    L2_vars = [x21, x22, x23, lambda2]
    L2_start = [sol2_grid_x[0], sol2_grid_x[1], L_sum-sol2_grid_x[0]-sol2_grid_x[1], 1]
    L2_focs = [simplify(diff(L2, var)) for var in L2_vars]
    sol2_equilibriums = nsolve(L2_focs, L2_vars, L2_start)
    X2_autarkie = sol2_equilibriums[0:3]
    U2_autarkie = U2(L2_subsistenz=X2_autarkie[2],C2_fisch_subsistenz=Y2_fisch(X2_autarkie[0]),C2_nuss_subsistenz=Y2_nuss(X2_autarkie[1]))
    print(f"  Lagrange:    L1_fisch={X1_autarkie[0]:4.1f}, L1_nuss={X1_autarkie[1]:4.1f}, L1_subsistenz={X1_autarkie[2]:4.1f}, Y1_fisch=C1_fisch={Y1_fisch(X1_autarkie[0]):5.2f}, Y1_nuss=C1_nuss={Y1_nuss(X1_autarkie[1]):5.2f}, U1={U1_autarkie:.2f}")
    print(f"  Lagrange:    L2_fisch={X2_autarkie[0]:4.1f}, L2_nuss={X2_autarkie[1]:4.1f}, L2_subsistenz={X2_autarkie[2]:4.1f}, Y2_fisch=C2_fisch={Y2_fisch(X2_autarkie[0]):5.2f}, Y2_nuss=C2_nuss={Y2_nuss(X2_autarkie[1]):5.2f}, U2={U2_autarkie:.2f}")
    # Numerische Lösung mittels scipy.optimize.minimize (gleiches Ergebnis wie Lagrange)
    sol1_equilibriums = scipy.optimize.minimize(fun = lambda x: -U1_fun(x), x0 = [sol1_grid_x[0], sol1_grid_x[1]], bounds=((0, 24), (0, 24)), method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
    sol2_equilibriums = scipy.optimize.minimize(fun = lambda x: -U2_fun(x), x0 = [sol2_grid_x[0], sol2_grid_x[1]], bounds=((0, 24), (0, 24)), method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
    if sol1_equilibriums.success and sol2_equilibriums.success:
        X1_autarkie = sol1_equilibriums.x
        X2_autarkie = sol2_equilibriums.x
        U1_autarkie = U1_fun(X1_autarkie)
        U2_autarkie = U2_fun(X2_autarkie)
        print(f"  Numerisch:   L1_fisch={X1_autarkie[0]:4.1f}, L1_nuss={X1_autarkie[1]:4.1f}, L1_subsistenz={L_sum-X1_autarkie[0]-X1_autarkie[1]:4.1f}, Y1_fisch=C1_fisch={Y1_fisch(X1_autarkie[0]):5.2f}, Y1_nuss=C1_nuss={Y1_nuss(X1_autarkie[1]):5.2f}, U1={U1_autarkie:.2f}")
        print(f"  Numerisch:   L2_fisch={X2_autarkie[0]:4.1f}, L2_nuss={X2_autarkie[1]:4.1f}, L2_subsistenz={L_sum-X2_autarkie[0]-X2_autarkie[1]:4.1f}, Y2_fisch=C2_fisch={Y2_fisch(X2_autarkie[0]):5.2f}, Y2_nuss=C2_nuss={Y2_nuss(X2_autarkie[1]):5.2f}, U2={U2_autarkie:.2f}")
    else:
        print(f"  ## WARNING: scipy.optimize.minimize failed, sol1_equilibriums={sol1_equilibriums}, sol2_equilibriums={sol2_equilibriums}")

    ##
    ## Fall 2: Robinson und Freitag verteilen die gemeinschaftlichen Erträge kooperativ nach festgelegten Verteilungsregeln. 
    ## Verteilungskriterien können z.B. sein:
    ## a) gleicher Nutzen, U1 = U2
    ## b) gleicher Konsum, C1_fisch_subsistenz = C2_fisch_subsistenz, C1_nuss_subsistenz = C2_nuss_subsistenz
    ## c) gleiche Arbeitszeit, L1_fisch + L1_nuss = L2_fisch + L2_nuss
    ## d) Nutzen proportional zum Arbeitseinsatz, Ui ∝ Li_fisch + Li_nuss ⇔ U1 / U2 = (L1_fisch + L1_nuss) / (L2_fisch + L2_nuss)
    ## e) Konsum proportional zum Arbeitseinsatz, Ci_fisch_subsistenz + Ci_nuss_subsistenz ∝ Li_fisch + Li_nuss ⇔ (C1_fisch_subsistenz + C1_nuss_subsistenz) / (C2_fisch_subsistenz + C2_nuss_subsistenz) = (L1_fisch + L1_nuss) / (L2_fisch + L2_nuss)
    ## Gemeinschaftlichen Erträge: 
    ## C1_fisch_subsistenz + C2_fisch_subsistenz = Y_fisch(L1_fisch, L2_fisch)
    ## C1_nuss_subsistenz + C2_nuss_subsistenz = Y_nuss(L1_nuss, L2_nuss)
    ## 6 Unbekannte: x[0] = L1_fisch, x[1] = L1_nuss, x[2] = L2_fisch, x[3] = L2_nuss, x[4] = C1_fisch_subsistenz/Y_fisch, x[5] = C1_nuss_subsistenz/Y_nuss
    ## Randbedingungen: x[0] + x[1] + L1_subsistenz = 24, x[2] + x[3] + L2_subsistenz = 24, 0 ≤ x[4] ≤ 1, 0 ≤ x[5] ≤ 1
    ##

    C1_fisch_fun = lambda x: x[4] * Y_fisch(x[0], x[2])       # x[4] = C1_fisch_subsistenz/Y_fisch
    C1_nuss_fun = lambda x: x[5] * Y_nuss(x[1], x[3])         # x[5] = C1_nuss_subsistenz/Y_nuss
    C2_fisch_fun = lambda x: (1 - x[4]) * Y_fisch(x[0], x[2]) # 1-x[4] = C2_fisch_subsistenz/Y_fisch
    C2_nuss_fun = lambda x: (1 - x[5]) * Y_nuss(x[1], x[3])   # 1-x[5] = C2_nuss_subsistenz/Y_nuss
    U1_fun = lambda x: U1(L1_subsistenz = L_sum - x[0] - x[1], C1_fisch_subsistenz = C1_fisch_fun(x), C1_nuss_subsistenz = C1_nuss_fun(x)) 
    U2_fun = lambda x: U2(L2_subsistenz = L_sum - x[2] - x[3], C2_fisch_subsistenz = C2_fisch_fun(x), C2_nuss_subsistenz = C2_nuss_fun(x))
    Uc_fun = lambda x: U1_delta * U2_delta if (U1_delta := U1_fun(x) - U1_autarkie) > 0 and (U2_delta := U2_fun(x) - U2_autarkie) > 0 else 0 # Nash-Produkt der Nutzenfunktionen, Maximierung von U_fun führt zu Pareto-effizienter Lösung mit positivem Handelsgewinn für beide
    Uc_cases = [ # Randbedingungen für die verschiedenen Verteilungskriterien in Fall 2
        # Fall 2a: Gemeinschaftliche Erträge, Verteilung nach gleichem Nutzen, U = U1_fun = U2_fun
        ("Verteilung der gemeinschaftlichen Erträge nach gleichem Nutzen", lambda x: (U1_fun(x) - U2_fun(x))), # Nebenbedingung: U1_fun(x) = U2_fun(x)
        # Fall 2b: Gemeinschaftliche Erträge, Verteilung nach gleichem Konsum, C1_fisch_subsistenz = C2_fisch_subsistenz, C1_nuss_subsistenz = C2_nuss_subsistenz, d.h. x[4] = 0.5, x[5] = 0.5
        ("Verteilung der gemeinschaftlichen Erträge nach gleichem Konsum", lambda x: abs(x[4] - 0.5) + abs(x[5] - 0.5)), # Nebenbedingung: x[4] = 0.5, x[5] = 0.5
        # Fall 2c: Gemeinschaftliche Erträge, Verteilung nach gleiche Arbeitszeit, L1_fisch + L1_nuss = L2_fisch + L2_nuss
        ("Verteilung der gemeinschaftlichen Erträge nach gleicher Arbeitszeit", lambda x: (x[0] + x[1] - x[2] - x[3])), # Nebenbedingung: L1_fisch + L1_nuss = L2_fisch + L2_nuss
        # Fall 2d: Gemeinschaftliche Erträge, Verteilung nach Nutzen proportional zum Arbeitseinsatz, Ui ∝ Li_fisch + Li_nuss ⇔ U1/U2 = (L1_fisch + L1_nuss)/(L2_fisch + L2_nuss)
        ("Verteilung der gemeinschaftlichen Erträge nach Nutzen proportional zum Arbeitseinsatz", lambda x: (U1_fun(x) / U2_fun(x) - (x[0] + x[1]) / (x[2] + x[3])) if U2_fun(x)>0 and (x[2]+x[3])>0 else 1e6), # Nebenbedingung: U1/U2 = (L1_fisch + L1_nuss)/(L2_fisch + L2_nuss)
        # Fall 2e: Gemeinschaftliche Erträge, Verteilung nach Konsum proportional zum Arbeitseinsatz, Ci_fisch_subsistenz + Ci_nuss_subsistenz ∝ Li_fisch + Li_nuss ⇔ (C1_fisch_subsistenz + C1_nuss_subsistenz) / (C2_fisch_subsistenz + C2_nuss_subsistenz) = (L1_fisch + L1_nuss) / (L2_fisch + L2_nuss)
        ("Verteilung der gemeinschaftlichen Erträge nach Konsum proportional zum Arbeitseinsatz", lambda x: ((C1_fisch_fun(x) + C1_nuss_fun(x)) / (C2_fisch_fun(x) + C2_nuss_fun(x)) - (x[0] + x[1]) / (x[2] + x[3])) if C2_fisch_fun(x)+C2_nuss_fun(x)>0 and (x[2]+x[3])>0 else 1e6), # Nebenbedingung: (C1_fisch_subsistenz + C1_nuss_subsistenz) / (C2_fisch_subsistenz + C2_nuss_subsistenz) = (L1_fisch + L1_nuss)
    ]
    for case in Uc_cases:
        title = case[0]
        constraints_fun = case[1]
        print(f"\nReine Subsistenzwirtschaft, 2 Personen, Güter sind Fische und Kokosnüsse, {title}:")
        # Lösung per grid_search (Plausibilitätscheck und Startwerte für die numerische Lösung)
        sol_grid_x, sol_grid_U = rcw_02_grid_search_iter(U_fun = Uc_fun, grid_x_start = [0, 0, 0, 0, 0, 0], grid_x_stop = [12, 12, 12, 12, 1, 1], grid_x_steps = [4, 4, 4, 4, 0.1, 0.1], step_type=GridStepType.DELTA_X, num_iter = 3, constraints_fun = lambda x: abs(constraints_fun(x)) <= 1e-2)
        if sol_grid_U <= 0:
            print(f"  grid-search: Nash-Produkt = {sol_grid_U} (kein Gewinn gegenüber Autarkie)")
            initial_guess = [9.0, 2.0, 3.0, 8.0, 0.5, 0.5]
        else:
            print(f"  grid-search: L1_fisch={sol_grid_x[0]:4.1f}, L1_nuss={sol_grid_x[1]:4.1f}, L1_subsistenz={L_sum-sol_grid_x[0]-sol_grid_x[1]:4.1f}, Y1_fisch={Y1_fisch(sol_grid_x[0]):5.2f}, C1_fisch={C1_fisch_fun(sol_grid_x):5.2f}, Y1_nuss={Y1_nuss(sol_grid_x[1]):5.2f}, C1_nuss={C1_nuss_fun(sol_grid_x):5.2f}, U1={U1_fun(sol_grid_x):.2f}")
            print(f"  grid-search: L2_fisch={sol_grid_x[2]:4.1f}, L2_nuss={sol_grid_x[3]:4.1f}, L2_subsistenz={L_sum-sol_grid_x[2]-sol_grid_x[3]:4.1f}, Y2_fisch={Y2_fisch(sol_grid_x[2]):5.2f}, C2_fisch={C2_fisch_fun(sol_grid_x):5.2f}, Y2_nuss={Y2_nuss(sol_grid_x[3]):5.2f}, C2_nuss={C2_nuss_fun(sol_grid_x):5.2f}, U2={U2_fun(sol_grid_x):.2f}")
            print(f"  grid-search: Nash-Produkt={Uc_fun(sol_grid_x):.3f}")
            initial_guess = sol_grid_x
        # Numerische Lösung mittels scipy.optimize.minimize
        sol_equilibrium = scipy.optimize.minimize(fun = lambda x: -(U1_fun(x) - U1_autarkie) * (U2_fun(x) - U2_autarkie), 
            x0 = initial_guess, bounds=((0, 24), (0, 24), (0, 24), (0, 24), (0, 1), (0, 1)), 
            constraints=[{'type': 'eq', 'fun': lambda x: constraints_fun(x)}, {"type": "ineq", "fun": lambda x: U1_fun(x) - U1_autarkie}, {"type": "ineq", "fun": lambda x: U2_fun(x) - U2_autarkie}], 
            method="SLSQP", options={"maxiter": 1000, "ftol": 1e-6})
        if sol_equilibrium.success:
            sol_x = sol_equilibrium.x
            print(f"  Numerisch:   L1_fisch={sol_x[0]:4.1f}, L1_nuss={sol_x[1]:4.1f}, L1_subsistenz={L_sum-sol_x[0]-sol_x[1]:4.1f}, Y1_fisch={Y1_fisch(sol_x[0]):5.2f}, C1_fisch={C1_fisch_fun(sol_x):5.2f}, Y1_nuss={Y1_nuss(sol_x[1]):5.2f}, C1_nuss={C1_nuss_fun(sol_x):5.2f}, U1={U1_fun(sol_x):.2f}")
            print(f"  Numerisch:   L2_fisch={sol_x[2]:4.1f}, L2_nuss={sol_x[3]:4.1f}, L2_subsistenz={L_sum-sol_x[2]-sol_x[3]:4.1f}, Y2_fisch={Y2_fisch(sol_x[2]):5.2f}, C2_fisch={C2_fisch_fun(sol_x):5.2f}, Y2_nuss={Y2_nuss(sol_x[3]):5.2f}, C2_nuss={C2_nuss_fun(sol_x):5.2f}, U2={U2_fun(sol_x):.2f}")
            print(f"  Numerisch:   Nash-Produkt={Uc_fun(sol_x):.3f}")
        else:
            print(f"  scipy.optimize.minimize failed (\"{sol_equilibrium.message}\")")

    ##
    ## Fall 3: Robinson und Freitag produzieren jeder für sich, betreiben aber Handel miteinander. Robinson tauscht T1_fisch seiner Fische gegen T2_nuss von Freitags Kokosnüssen. 
    ## tau1_fisch = T1_fisch / Y1_fisch, tau2_nuss = T2_nuss / Y2_nuss <=> T1_fisch = tau1_fisch * Y1_fisch, T2_nuss = tau2_nuss * Y2_nuss
    ## C1_fisch_subsistenz  = Y1_fisch - T1_fisch = (1 - tau1_fisch) * Y1_fisch
    ## C2_fisch_subsistenz  = Y2_fisch + T1_fisch = Y2_fisch + tau1_fisch * Y1_fisch
    ## C1_nuss_subsistenz  = Y1_nuss + T2_nuss = Y1_nuss + tau2_nuss * Y2_nuss
    ## C2_nuss_subsistenz  = Y2_nuss - T2_nuss = (1 - tau2_nuss) * Y2_nuss
    ## mit den Nebenbedingungen: 0 ≤ tau1_fisch ≤ 1, 0 ≤ tau2_nuss ≤ 1
    ##
    
    # Fall 3, Warentausch, 6 Unbekannte: x[0] = L1_fisch, x[1] = L1_nuss, x[2] = L2_fisch, x[3] = L2_nuss, x[4] = tau1_fisch, x[5] = tau2_nuss
    L1_subsistenz = lambda x: L_sum - x[0] - x[1]               # Anzahl der von Robinson für Subsistenz aufgewendeten Stunden
    L2_subsistenz = lambda x: L_sum - x[2] - x[3]               # Anzahl der von Freitag für Subsistenz aufgewendeten Stunden
    C1_fisch = lambda x: (1 - x[4]) * Y1_fisch(x[0])            # Anzahl der von Robinson konsumierten Fische
    C2_fisch = lambda x: Y2_fisch(x[2]) + x[4] * Y1_fisch(x[0]) # Anzahl der von Freitag konsumierten Fische
    C1_nuss = lambda x: Y1_nuss(x[1]) + x[5] * Y2_nuss(x[3])    # Anzahl der von Robinson konsumierten Kokosnüsse
    C2_nuss = lambda x: (1 - x[5]) * Y2_nuss(x[3])              # Anzahl der von Freitag konsumierten Kokosnüsse
    T1_fisch = lambda x: x[4] * Y1_fisch(x[0])                  # Anzahl der von Robinson getauschten Fische
    T2_nuss = lambda x: x[5] * Y2_nuss(x[3])                    # Anzahl der von Freitag getauschten Kokosnüsse
    r_fisch_nuss = lambda x: T1_fisch(x) / T2_nuss(x) if T2_nuss(x)>0 else np.nan # Tauschverhältnis, getauschte Fische pro getauschte Kokosnuss
    tau_fisch_nuss = lambda x: (T1_fisch(x) + T2_nuss(x)) / (Y1_fisch(x[0]) + Y1_nuss(x[1]) + Y2_fisch(x[2]) + Y2_nuss(x[3]))   # Handelsintensität, Anteil der gehandelten Güter am Gesamtertrag
    U1_fun = lambda x: U1(L1_subsistenz = L1_subsistenz(x), C1_fisch_subsistenz = C1_fisch(x), C1_nuss_subsistenz = C1_nuss(x)) # Robinsons Nutzen
    U2_fun = lambda x: U2(L2_subsistenz = L2_subsistenz(x), C2_fisch_subsistenz = C2_fisch(x), C2_nuss_subsistenz = C2_nuss(x)) # Freitags Nutzen
    U_fun = lambda x: U1_delta * U2_delta if (U1_delta := U1_fun(x) - U1_autarkie) > 0 and (U2_delta := U2_fun(x) - U2_autarkie) > 0 else 0 # Nash-Produkt der Nutzenfunktionen, Maximierung von U_fun führt zu Pareto-effizienter Lösung mit positivem Handelsgewinn für beide
    # Lösung per grid_search (Plausibilitätscheck und Startwerte für die numerische Lösung)
    print(f"\nReine Subsistenzwirtschaft, 2 Personen, Güter sind Fische und Kokosnüsse, Warentausch:")
    sol_grid_x, sol_grid_U = rcw_02_grid_search_iter(U_fun = U_fun, grid_x_start = [0, 0, 0, 0, 0, 0], grid_x_stop = [24, 24, 24, 24, 1, 1], grid_x_steps = [4, 4, 4, 4, 0.1, 0.1], step_type=GridStepType.DELTA_X, num_iter = 3, constraints_fun = None)
    if sol_grid_U <= 0:
        print(f"  grid-search: Nash-Produkt = {sol_grid_U}, kein Gewinn gegenüber Autarkie")
    else:
        print(f"  grid-search: L1_fisch={sol_grid_x[0]:4.1f}, L1_nuss={sol_grid_x[1]:4.1f}, L1_subsistenz={L1_subsistenz(sol_grid_x):4.1f}, Y1_fisch={Y1_fisch(sol_grid_x[0]):5.2f}, C1_fisch={C1_fisch(sol_grid_x):5.2f}, Y1_nuss={Y1_nuss(sol_grid_x[1]):5.2f}, C1_nuss={C1_nuss(sol_grid_x):5.2f}, U1={U1_fun(sol_grid_x):.2f}")
        print(f"  grid-search: L2_fisch={sol_grid_x[2]:4.1f}, L2_nuss={sol_grid_x[3]:4.1f}, L2_subsistenz={L2_subsistenz(sol_grid_x):4.1f}, Y2_fisch={Y2_fisch(sol_grid_x[2]):5.2f}, C2_fisch={C2_fisch(sol_grid_x):5.2f}, Y2_nuss={Y2_nuss(sol_grid_x[3]):5.2f}, C2_nuss={C2_nuss(sol_grid_x):5.2f}, U2={U2_fun(sol_grid_x):.2f}")
        print(f"  grid-search: τ1_fisch={sol_grid_x[4]:.2f}, T1_fisch={T1_fisch(sol_grid_x):.2f}, τ2_nuss={sol_grid_x[5]:.2f}, T2_nuss={T2_nuss(sol_grid_x):.2f}, Nash-Produkt={U_fun(sol_grid_x):.3f}")
    # Numerische Lösung mittels scipy.optimize.minimize; statt (U1 - U1_autarkie) * (U2 - U2_autarkie) zu maximieren wird -log(U1 - U1_autarkie) - log(U2 - U2_autarkie) minimiert (gleiches Ergebnis, aber bessere numerische Stabilität)
    sol_bounds = ((0, 24), (0, 24), (0, 24), (0, 24), (0, 1), (0, 1))
    sol_equilibrium = scipy.optimize.minimize(fun = lambda x: -log(U1_fun(x) - U1_autarkie) - log(U2_fun(x) - U2_autarkie), # fun = lambda x: -(U1_fun(x) - U1_autarkie) * (U2_fun(x) - U2_autarkie)
        x0 = sol_grid_x, bounds = sol_bounds, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9},
        constraints=[{"type": "ineq", "fun": lambda x: U1_fun(x) - U1_autarkie}, {"type": "ineq", "fun": lambda x: U2_fun(x) - U2_autarkie}])
    if sol_equilibrium.success:
        sol_x = sol_equilibrium.x
        sol_U = U_fun(sol_x)
        print(f"  Numerisch:   L1_fisch={sol_x[0]:4.1f}, L1_nuss={sol_x[1]:4.1f}, L1_subsistenz={L1_subsistenz(sol_x):4.1f}, Y1_fisch={Y1_fisch(sol_x[0]):5.2f}, C1_fisch={C1_fisch(sol_x):5.2f}, Y1_nuss={Y1_nuss(sol_x[1]):5.2f}, C1_nuss={C1_nuss(sol_x):5.2f}, U1={U1_fun(sol_x):.2f}")
        print(f"  Numerisch:   L2_fisch={sol_x[2]:4.1f}, L2_nuss={sol_x[3]:4.1f}, L2_subsistenz={L2_subsistenz(sol_x):4.1f}, Y2_fisch={Y2_fisch(sol_x[2]):5.2f}, C2_fisch={C2_fisch(sol_x):5.2f}, Y2_nuss={Y2_nuss(sol_x[3]):5.2f}, C2_nuss={C2_nuss(sol_x):5.2f}, U2={U2_fun(sol_x):.2f}")
        print(f"  Numerisch:   τ1_fisch={sol_x[4]:.2f}, T1_fisch={T1_fisch(sol_x):.2f}, τ2_nuss={sol_x[5]:.2f}, T2_nuss={T2_nuss(sol_x):.2f}, T1_fisch/T2_nuss={r_fisch_nuss(sol_x):.2f}, T/Y={tau_fisch_nuss(sol_x):.2f}, Nash-Produkt={sol_U:.3f}")
        # Stabilitätscheck: Kleine Abweichungen von der gefundenen Lösung sollten zu einem kleinen Rückgang des Nash-Produkts führen
        for check_iter in range(1000):
            check_x = sol_x + np.random.uniform(0, 1.0e-3, size=len(sol_x))
            check_U = U_fun(check_x)
            if check_U > 0 and (check_U < sol_U - 1.0e-2 or check_U > sol_U + 1.0e-6):
                print(f"  Numerisch Lösung möglicherweise instabil: Nash-Produkt={sol_U:.6f}, Nash-Produkt(x={check_x}) = {check_U:.6f}, delta={sol_U-check_U:.6f}")
                break
        # Stabilitätscheck: Zufällige Startpunkte sollten nicht zu besseren Lösungen führen
        for check_iter in range(100):
            check_x0 = [ np.random.uniform(0,b[1]) for b in sol_bounds ]
            check_sol = scipy.optimize.minimize(fun = lambda x: -(U1_fun(x) - U1_autarkie) * (U2_fun(x) - U2_autarkie),
                x0 = check_x0, bounds = sol_bounds, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9},
                constraints=[{"type": "ineq", "fun": lambda x: U1_fun(x) - U1_autarkie}, {"type": "ineq", "fun": lambda x: U2_fun(x) - U2_autarkie}])
            if check_sol.success and (check_U := U_fun(check_sol.x)) > sol_U + 1.0e-6:
                print(f"  Numerisch Lösung möglicherweise instabil: Nash-Produkt={sol_U:.6f} mit x0={sol_grid_x}, Nash-Produkt={check_U:.6f} mit zufälligem x0={check_x0}")
                break

    else:
        print(f"  scipy.optimize.minimize failed (\"{sol_equilibrium.message}\")")

# Robinson-Crusoe-Wirtschaft mit 2 Personen (Robinson und Freitag) mit reiner Subsistenzwirtschaft.
# Die Produktion besteht aus Fischen, Kokosnüssen und Brennholz, die vollständig zur Subsistenz konsumiert werden.
# Intertemporale Optimierung: die diskontierte Nutzenfunktion $U_sum = \sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
# t_max = 20 bedeutet: sum_{t=0}^{∞} wird ersetzt durch sum_{t=0}^{t_max} mit t_max = 20, da δ^t U(t) mit zunehmendem t immer kleiner wird (delta < 1). 
def rcw_02_subsistence_fish_nuts_wood(t_max = 20):
    
    # Beispielparameter
    alpha_fisch = 0.5 # Elastizität von Arbeit in der Fischproduktion
    alpha_nuss = 0.4  # Elastizität von Arbeit in der Kokosnussproduktion
    alpha_holz = 1    # Elastizität von Arbeit in der Brennholzproduktion, doppelte Holzfällerarbeit Li_holz produziert doppelt so viel Brennholz Yi_holz
    A1_fisch = 20 / (12**alpha_fisch) # Robinsons Produktivität bei der Fischproduktion Y1_fisch = A1_fisch * L1_fisch^alpha_fisch, Robinson fängt 20 Fische in 12 Stunden
    A2_fisch = 10 / (12**alpha_fisch) # Freitags Produktivität bei der Fischproduktion Y2_fisch = A2_fisch * L2_fisch^alpha_fisch, Freitag fängt 10 Fische in 12 Stunden
    A1_nuss = 15 / (12**alpha_nuss)   # Robinsons Produktivität bei der Kokosnussproduktion Y1_nuss = A1_nuss * L1_nuss^alpha_nuss, Robinson erntet 15 Kokosnüsse in 12 Stunden
    A2_nuss = 30 / (12**alpha_nuss)   # Freitags Produktivität bei der Kokosnussproduktion Y2_nuss = A2_nuss * L2_nuss^alpha_nuss, Freitag erntet 30 Kokosnüsse in 12 Stunden
    A1_holz = 1 / (1**alpha_holz)     # Robinsons Produktivität bei der Brennholzproduktion Y1_holz = A1_holz * L1_holz, Robinson benötigt 1 Stunde, um 1 Baum zu fällen und zu Brennholz zu verarbeiten
    A2_holz = 1 / (2**alpha_holz)     # Freitags Produktivität bei der Brennholzproduktion Y1_holz = A1_holz * L1_holz, Freitag benötigt 2 Stunden, um 1 Baum zu fällen und zu Brennholz zu verarbeiten
    K_wald_max = 1000 # Maximale Größe des Waldes, z.B. 1000 Bäume
    g_wald = 0.01 # Natürliche Wachstumsrate des Waldes, z.B. 1% pro Zeiteinheit
    delta = 0.5 # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ mit Diskontfaktor delta wird maximiert.

    # Shortcut für alle Elemente in _list >= 0
    ge0 = lambda _list : all(_elem >= 0 for _elem in _list) # true, wenn alle Elemente in _list >= 0 sind

    # Cobb-Douglas-Produktionsfunktionen für Robinson (Y1_fisch, Y1_nuss, Y1_holz) und Freitag (Y2_fisch, Y2_nuss, Y2_holz)
    Y1_fisch = lambda L1_fisch: A1_fisch * L1_fisch ** alpha_fisch if ge0([L1_fisch]) else np.nan
    Y2_fisch = lambda L2_fisch: A2_fisch * L2_fisch ** alpha_fisch if ge0([L2_fisch]) else np.nan
    Y_fisch = lambda L1_fisch, L2_fisch: Y1_fisch(L1_fisch) + Y2_fisch(L2_fisch)
    Y1_nuss = lambda L1_nuss: A1_nuss * L1_nuss ** alpha_nuss if ge0([L1_nuss]) else np.nan
    Y2_nuss = lambda L2_nuss: A2_nuss * L2_nuss ** alpha_nuss if ge0([L2_nuss]) else np.nan
    Y_nuss = lambda L1_nuss, L2_nuss: Y1_nuss(L1_nuss) + Y2_nuss(L2_nuss)
    Y1_holz = lambda L1_holz: A1_holz * L1_holz ** alpha_holz if ge0([L1_holz]) else np.nan
    Y2_holz = lambda L2_holz: A2_holz * L2_holz ** alpha_holz if ge0([L2_holz]) else np.nan
    Y_holz = lambda L1_holz, L2_holz: Y1_holz(L1_holz) + Y2_holz(L2_holz)

    # Nutzenfunktionen für Robinson (U1) und Freitag (U2):
    # Ui_subsistenz = Ai_subsistenz * Li_subsistenz^αi_subsistenz * Ci_fisch_subsistenz^βi_subsistenz * Ci_nuss_subsistenz^γi_subsistenz
    alpha_subsistenz = 2 # Subsistenz mit höherer Elastizität, hier vereinfacht αi_subsistenz = βi_subsistenz = γi_subsistenz = 2
    alpha_warme = 0.5 # Elastizität des Nutzens aus Brennholz, U_holz = A_warme * Y_holz^α_wärme, z.B. α_wärme = 0.5 für abnehmenden Grenznutzen (doppelte Menge Brennholz führt nicht zu doppeltem Nutzen)
    A1_subsistenz = 1 / ((10**alpha_subsistenz) * ((20)**alpha_subsistenz) * (10**alpha_subsistenz)) # Normierung: U1 = 1 bei L_subsistenz = 10 Stunden sowie 20 konsumierten Fischen und 10 konsumierten Kokosnüssen
    A2_subsistenz = 1 / ((8**alpha_subsistenz) * ((16)**alpha_subsistenz) * (15**alpha_subsistenz))  # Normierung: U2 = 1 bei L_subsistenz =  8 Stunden sowie 16 konsumierten Fischen und 15 konsumierten Kokosnüssen
    U1_subsistenz = lambda L1_subsistenz, C1_fisch_subsistenz, C1_nuss_subsistenz: A1_subsistenz * (L1_subsistenz**alpha_subsistenz) * (C1_fisch_subsistenz**alpha_subsistenz) * (C1_nuss_subsistenz**alpha_subsistenz) if ge0([L1_subsistenz, C1_fisch_subsistenz, C1_nuss_subsistenz]) else np.nan
    U2_subsistenz = lambda L2_subsistenz, C2_fisch_subsistenz, C2_nuss_subsistenz: A2_subsistenz * (L2_subsistenz**alpha_subsistenz) * (C2_fisch_subsistenz**alpha_subsistenz) * (C2_nuss_subsistenz**alpha_subsistenz) if ge0([L2_subsistenz, C2_fisch_subsistenz, C2_nuss_subsistenz]) else np.nan
    # Nutzen des gemeinsamen Lagerfeuers:
    # U1_holz = U2_holz = A_warme * (C1_holz + C2_holz)^α_U_holz = A_warme * (Y1_holz + Y2_holz)^α_U_holz = A_warme * (A1_holz * L1_holz + A2_holz * L2_holz)^α_U_holz
    U1_holz = lambda A_warme, L1_holz, L2_holz: A_warme * (Y_holz(L1_holz, L2_holz)**alpha_warme) if ge0([L1_holz, L2_holz]) else np.nan
    U2_holz = lambda A_warme, L1_holz, L2_holz: U1_holz(A_warme, L1_holz, L2_holz) # das gemeinsame Lagerfeuer wärmt beide gleich, also gleicher Nutzen

    # Zeitrestriktion auf 24 Stunden: L1_subsistenz = L_sum - L1_fisch - L1_nuss - L1_holz, L2_subsistenz = L_sum - L2_fisch - L2_nuss - L2_holz
    L_sum = 24

    # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{t_max}δ^t U(t)$ wird maximiert.
    # Tauschhandel: Robinson tauscht T1_fisch=tau1_fisch*Y1_fisch seiner Fischproduktion plus T1_holz=tau1_holz*Y1_holz seiner Holzproduktion gegen T2_nuss=tau2_nuss*Y2_nuss von Freitags Kokosnüssen.
    # 9 Unbekannte: x[0] = L1_fisch, x[1] = L1_nuss, x[2] = L1_holz, x[3] = L2_fisch, x[4] = L2_nuss, x[5] = L2_holz, x[6] = tau1_fisch, x[7] = tau2_nuss, x[8] = tau1_holz
    L1_fisch, L1_nuss, L1_holz, L2_fisch, L2_nuss, L2_holz, tau1_fisch, tau2_nuss, tau1_holz = lambda x: x[0], lambda x: x[1], lambda x: x[2], lambda x: x[3], lambda x: x[4], lambda x: x[5], lambda x: x[6], lambda x: x[7], lambda x: x[8]
    L1_subsistenz = lambda x: L_sum - L1_fisch(x) - L1_nuss(x) - L1_holz(x)       # Anzahl der von Robinson für Subsistenz aufgewendeten Stunden
    L2_subsistenz = lambda x: L_sum - L2_fisch(x) - L2_nuss(x) - L2_holz(x)       # Anzahl der von Freitag für Subsistenz aufgewendeten Stunden
    T1_fisch = lambda x: tau1_fisch(x) * Y1_fisch(L1_fisch(x))                    # Menge der von Robinson getauschten Fische
    T2_nuss = lambda x: tau2_nuss(x) * Y2_nuss(L2_nuss(x))                        # Menge der von Freitag getauschten Kokosnüsse
    T1_holz = lambda x: tau1_holz(x) * Y1_holz(L1_holz(x))                        # Menge des von Robinson getauschten Brennholzes
    C1_fisch = lambda x: Y1_fisch(L1_fisch(x)) - T1_fisch(x)                      # Menge der von Robinson konsumierten Fische
    C1_nuss = lambda x: Y1_nuss(L1_nuss(x)) + T2_nuss(x)                          # Menge der von Robinson konsumierten Kokosnüsse
    C1_holz = lambda x: Y1_holz(L1_holz(x)) - T1_holz(x)                          # Menge des von Robinson konsumierten Brennholzes
    C2_fisch = lambda x: Y2_fisch(L2_fisch(x)) + T1_fisch(x)                      # Menge der von Freitag konsumierten Fische
    C2_nuss = lambda x: Y2_nuss(L2_nuss(x)) - T2_nuss(x)                          # Menge der von Freitag konsumierten Kokosnüsse
    C2_holz = lambda x: Y2_holz(L2_holz(x)) + T1_holz(x)                          # Menge des von Freitag konsumierten Brennholzes
    r_fisch_nuss = lambda x: T1_fisch(x) / T2_nuss(x) if T2_nuss(x)>0 else np.nan # Tauschverhältnis, getauschte Fische pro getauschte Kokosnuss
    r_holz_nuss = lambda x: T1_holz(x) / T2_nuss(x) if T2_nuss(x)>0 else np.nan   # Tauschverhältnis, getauschtes Brennholz pro getauschte Kokosnuss
    trade_intensity = lambda x: (T1_fisch(x) + T2_nuss(x) + T1_holz(x)) / (Y1_fisch(L1_fisch(x)) + Y1_nuss(L1_nuss(x)) + Y1_holz(L1_holz(x)) + Y2_fisch(L2_fisch(x)) + Y2_nuss(L2_nuss(x)) + Y2_holz(L2_holz(x))) # Handelsintensität, Anteil der gehandelten Güter am Gesamtertrag
    U1_fun = lambda x, A_warme: U1_subsistenz(L1_subsistenz=L1_subsistenz(x), C1_fisch_subsistenz=C1_fisch(x), C1_nuss_subsistenz=C1_nuss(x)) + U1_holz(A_warme, L1_holz(x), L2_holz(x)) # Robinsons Nutzenfunktion (Subsistenz + Lagerfeuer)
    U2_fun = lambda x, A_warme: U2_subsistenz(L2_subsistenz=L2_subsistenz(x), C2_fisch_subsistenz=C2_fisch(x), C2_nuss_subsistenz=C2_nuss(x)) + U2_holz(A_warme, L1_holz(x), L2_holz(x)) # Freitags Nutzenfunktion (Subsistenz + Lagerfeuer)

    # Autarkie-Nutzen: Robinson und Freitag produzieren und konsumieren jeweils nur für sich selbst. 
    # Jeder maximiert seine eigene Nutzenfunktion U_i unabhängig vom anderen. Kein Handel, tau1_fisch = tau2_nuss = tau1_holz = 0
    # 3 Unbekannte: x[0] = L1_fisch, x[1] = L1_nuss, x[2] = L1_holz für Robinson bzw. x[0] = L2_fisch, x[1] = L2_nuss, x[2] = L2_holz für Freitag
    U1_fun_autarkie = lambda x, A_warme: U1_subsistenz(L1_subsistenz=L_sum-x[0]-x[1]-x[2], C1_fisch_subsistenz=Y1_fisch(x[0]), C1_nuss_subsistenz=Y1_nuss(x[1])) + U1_holz(A_warme,x[2],0) # Robinsons Autarkie-Nutzen ohne Handel (Subsistenz + Brennholz)
    U2_fun_autarkie = lambda x, A_warme: U2_subsistenz(L2_subsistenz=L_sum-x[0]-x[1]-x[2], C2_fisch_subsistenz=Y2_fisch(x[0]), C2_nuss_subsistenz=Y2_nuss(x[1])) + U2_holz(A_warme,0,x[2]) # Freitags Autarkie-Nutzen ohne Handel (Subsistenz + Brennholz)

    # Nutzenfaktor für die Wärme des Lagerfeuers: Ui_holz = A_warme * (C1_holz + C2_holz)^α_U_holz
    # Die dominante Strategie ist L1_holz=L2_holz=0 mit dem Ergebnis U1_holz=U2_holz=0, d.h. kein Lagerfeuer (Dilemma: jeder möchte möglichst wenig zum gemeinsamen Lagerfeuer beizutragen).
    # Der Fall 2/3 < A_warme < 1 ist aber besonders interessant: In dieser Zone entsteht ein nicht-pareto-effizientes Nash-Gleichgewicht, d.h. beide können sich verbessern, wenn sie die dominante Strategie (L1_holz=L2_holz=0)
    # verlassen und gemeinsam ein Lagerfeuer vereinbaren (L1_holz>0, L2_holz>0 führt zu U1_holz>0 und U2_holz>0).
    # In allen anderen Fällen ist das Nash-Gleichgewicht (L1_holz=L2_holz=0, U1_holz=U2_holz=0) Pareto-effizient, d.h. es existiert keine andere Kombination, bei der sich beide verbessern.
    print(f"Reine Subsistenzwirtschaft, 2 Personen, Güter sind Fische, Kokosnüsse und Holz:")
    for A_warme in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: # [0.5, 0.9, 4]:

        # Autarkie-Nutzen
        # sol1_autarkie_grid_x, sol1_autarkie_grid_U = rcw_02_grid_search_iter(U_fun = lambda x: U1_fun_autarkie(x, A_warme), grid_x_start = [0, 0, 0], grid_x_stop = [24, 24, 24], grid_x_steps = [1, 1, 1], step_type=GridStepType.DELTA_X, num_iter = 3, constraints_fun = None)
        # sol2_autarkie_grid_x, sol2_autarkie_grid_U = rcw_02_grid_search_iter(U_fun = lambda x: U2_fun_autarkie(x, A_warme), grid_x_start = [0, 0, 0], grid_x_stop = [24, 24, 24], grid_x_steps = [1, 1, 1], step_type=GridStepType.DELTA_X, num_iter = 3, constraints_fun = None)
        # print(f"  grid-search: X1_autarkie={fmt(sol1_autarkie_grid_x,'{:4.2f}',',')}, X2_autarkie={fmt(sol2_autarkie_grid_x,'{:4.2f}',',')}, U1_autarkie={sol1_autarkie_grid_U:.2f}, U2_autarkie={sol2_autarkie_grid_U:.2f}")
        sol1_autarkie = scipy.optimize.minimize(fun = lambda x: -U1_fun_autarkie(x, A_warme), x0 = [5, 5, 1], bounds=((0, 24), (0, 24), (0, 24)), method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
        sol2_autarkie = scipy.optimize.minimize(fun = lambda x: -U2_fun_autarkie(x, A_warme), x0 = [5, 5, 1], bounds=((0, 24), (0, 24), (0, 24)), method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
        if sol1_autarkie.success and sol2_autarkie.success:
            X1_autarkie = sol1_autarkie.x
            X2_autarkie = sol2_autarkie.x
            U1_autarkie = U1_fun_autarkie(X1_autarkie, A_warme)
            U2_autarkie = U2_fun_autarkie(X2_autarkie, A_warme)
            print(f"  Autarkienutzen (numerisch): A_warme={A_warme}, X1_autarkie={fmt(X1_autarkie,'{:4.2f}',',')}, X2_autarkie={fmt(X2_autarkie,'{:4.2f}',',')}, U1_autarkie={U1_autarkie:.2f}, U2_autarkie={U2_autarkie:.2f}")
        else:
            print(f"  Autarkienutzen (numerisch): A_warme={A_warme}, scipy.optimize.minimize failed (\"{sol1_autarkie.message}\", \"{sol2_autarkie.message}\")")
            continue

        def U_sum(x, U_fun):
            U_sum = 0
            K_wald = K_wald_max
            for t in range(t_max):

                # Randbedingung: Robinson und Freitag können nicht mehr Holz fällen als Bäume existieren:
                # Y_holz <= K_wald
                # if Y_holz < 0: # Strafterm (negativer Nutzen) wenn Ressourcenrestriktion verletzt wird
                #     return -1e6 + Y_holz

                # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
                U_sum += (delta**t) * U_fun(x, A_warme)

                # Der Wald wird als begrenztes Kapital modelliert
                K_wald_left = K_wald - Y_holz(L1_holz(x), L2_holz(x))
                K_wald = K_wald_left + (g_wald * K_wald_left) * (1 - K_wald_left / K_wald_max)
            return U_sum

        nash_product = lambda x: (U_sum(x, U1_fun) - U1_autarkie) * (U_sum(x, U2_fun) - U2_autarkie) # Nash-Produkt der Nutzenfunktionen, Maximierung führt zu Pareto-effizienter Lösung mit positivem Handelsgewinn für beide

    # # Fall 3, Warentausch, 6 Unbekannte: x[0] = L1_fisch, x[1] = L1_nuss, x[2] = L2_fisch, x[3] = L2_nuss, x[4] = tau1_fisch, x[5] = tau2_nuss
    # U_fun = lambda x: U1_delta * U2_delta if (U1_delta := U1_fun(x) - U1_autarkie) > 0 and (U2_delta := U2_fun(x) - U2_autarkie) > 0 else 0 # Nash-Produkt der Nutzenfunktionen, Maximierung von U_fun führt zu Pareto-effizienter Lösung mit positivem Handelsgewinn für beide
    # # Lösung per grid_search (Plausibilitätscheck und Startwerte für die numerische Lösung)
    # print(f"\nReine Subsistenzwirtschaft, 2 Personen, Warentausch:")
    # sol_grid_x, sol_grid_U = rcw_02_grid_search_iter(U_fun = U_fun, grid_x_start = [0, 0, 0, 0, 0, 0], grid_x_stop = [24, 24, 24, 24, 1, 1], grid_x_steps = [4, 4, 4, 4, 0.1, 0.1], step_type=GridStepType.DELTA_X, num_iter = 3, constraints_fun = None)
    # if sol_grid_U <= 0:
    #     print(f"  grid-search: Nash-Produkt = {sol_grid_U}, kein Gewinn gegenüber Autarkie")
    # else:
    #     print(f"  grid-search: L1_fisch={sol_grid_x[0]:4.1f}, L1_nuss={sol_grid_x[1]:4.1f}, L1_subsistenz={L1_subsistenz(sol_grid_x):4.1f}, Y1_fisch={Y1_fisch(sol_grid_x[0]):5.2f}, C1_fisch={C1_fisch(sol_grid_x):5.2f}, Y1_nuss={Y1_nuss(sol_grid_x[1]):5.2f}, C1_nuss={C1_nuss(sol_grid_x):5.2f}, U1={U1_fun(sol_grid_x):.2f}")
    #     print(f"  grid-search: L2_fisch={sol_grid_x[2]:4.1f}, L2_nuss={sol_grid_x[3]:4.1f}, L2_subsistenz={L2_subsistenz(sol_grid_x):4.1f}, Y2_fisch={Y2_fisch(sol_grid_x[2]):5.2f}, C2_fisch={C2_fisch(sol_grid_x):5.2f}, Y2_nuss={Y2_nuss(sol_grid_x[3]):5.2f}, C2_nuss={C2_nuss(sol_grid_x):5.2f}, U2={U2_fun(sol_grid_x):.2f}")
    #     print(f"  grid-search: τ1_fisch={sol_grid_x[4]:.2f}, T1_fisch={T1_fisch(sol_grid_x):.2f}, τ2_nuss={sol_grid_x[5]:.2f}, T2_nuss={T2_nuss(sol_grid_x):.2f}, Nash-Produkt={U_fun(sol_grid_x):.3f}")
    # # Numerische Lösung mittels scipy.optimize.minimize; statt (U1 - U1_autarkie) * (U2 - U2_autarkie) zu maximieren wird -log(U1 - U1_autarkie) - log(U2 - U2_autarkie) minimiert (gleiches Ergebnis, aber bessere numerische Stabilität)
    # sol_bounds = ((0, 24), (0, 24), (0, 24), (0, 24), (0, 1), (0, 1))
    # sol_equilibrium = scipy.optimize.minimize(fun = lambda x: -log(U1_fun(x) - U1_autarkie) - log(U2_fun(x) - U2_autarkie), # fun = lambda x: -(U1_fun(x) - U1_autarkie) * (U2_fun(x) - U2_autarkie)
    #     x0 = sol_grid_x, bounds = sol_bounds, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9},
    #     constraints=[{"type": "ineq", "fun": lambda x: U1_fun(x) - U1_autarkie}, {"type": "ineq", "fun": lambda x: U2_fun(x) - U2_autarkie}])
    # if sol_equilibrium.success:
    #     sol_x = sol_equilibrium.x
    #     sol_U = U_fun(sol_x)
    return

if __name__ == "__main__":

    # Robinson-Crusoe-Wirtschaft mit 2 Personen (Robinson und Freitag) mit reiner Subsistenzwirtschaft.
    # Die Produktion besteht aus Fischen und Kokosnüssen und wird vollständig zur Subsistenz konsumiert.
    # rcw_02_subsistence_fish_nuts()

    # Produktion von Fischen, Kokosnüssen und Brennholz, die vollständig zur Subsistenz konsumiert werden.
    rcw_02_subsistence_fish_nuts_wood()
    
    plt.show()
    print(f"")
