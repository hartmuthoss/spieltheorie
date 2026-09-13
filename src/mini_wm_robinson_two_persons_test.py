"""
Hilfs- und Testfunktionen für Mini-WM Teil 2 (Robinson-Crusoe-Wirtschaft mit 2 Personen)
Erläuterungen siehe  MiniWM_Teil02_RobinsonFreitag.md.
"""
from enum import IntEnum
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
import scipy # pip install scipy
import scipy.optimize
import sympy as sp # pip install sympy
import time
from typing import Literal

# Sucht das Maximum der Funktion opt_fun(x), indem alle Achsen i sequentiell von xmin[i] bis xmax[i] bei sonst unverändertem x0 mit gegebenen Schrittanzahl steps durchlaufen werden. Optional mit Plot der Funktionswerte.
def axes_search(opt_fun, x0, xmin, xmax, steps, do_plot=False, title="axes_search"):
    assert(len(x0) == len(xmin) and len(xmin) == len(xmax) and len(x0) > 0)
    max_value = None
    opt_x = None
    if do_plot:
        plt.figure(figsize=(12,5))
    for i in range(len(x0)):
        x_values_axis_i = np.array([np.nan] * steps, dtype=float)
        fun_values_axis_i = np.array([np.nan] * steps, dtype=float)
        for n, xn in enumerate(np.linspace(xmin[i], xmax[i], steps)):
            x = np.copy(x0)
            x[i] = xn
            fun_value = opt_fun(x)
            if fun_value is not None and np.isfinite(fun_value):
                fun_values_axis_i[n] = fun_value
                x_values_axis_i[n] = (x[i] - xmin[i]) / (xmax[i] - xmin[i])
                if max_value is None or max_value < fun_values_axis_i[n]:
                    max_value = fun_values_axis_i[n]
                    opt_x = np.copy(x)
        if do_plot:
            plt.plot(x_values_axis_i[np.isfinite(x_values_axis_i)], fun_values_axis_i[np.isfinite(fun_values_axis_i)], label = f"axis {i+1}")
    if do_plot:
        plt.xlabel(f"(x-xmin)/(xmax-xmin)")
        plt.ylabel(f"fun(x)")
        plt.legend()
        plt.title(f"{title}:\nx0={fmt(x0,'{:.3f}')}, opt_x={fmt(opt_x,'{:.3f}')}, max_value={max_value:.2f}")
        plt.show(block=False)
    return opt_x, max_value

# Maximierung des diskontierten Autarkienutzens mit Suche nach dem optimalen Startwert x0, Rückgabe x, Nutzen U, diskontierter Nutzen U_diskont und constraints unter Autarkie
def search_Ui_autarkie_diskont(i: Literal[1, 2], delta_i: float, t_start: float, verbose: int = 1):
    from mini_wm_robinson_two_persons import calc_Ui_autarkie_diskont, fmt, maximize_Ui_autarkie_diskont, min_constraints
    # Berechnung der diskontierten Autarkienutzen V1_autarkie, V2_autarkie mit Startwerten x0_autarkie:
    # x0_autarkie = np.array([L_max/4, L_max/4, Li_subsistenz_min, 1e-3, 1e-3, 1.0, 1.0], dtype=np.float64)
    # xmin_autarkie = np.array([ 0,  0,  8, 0.0, 0.0, 0, 0], dtype=np.float64)
    # xmax_autarkie = np.array([12, 12, 12, 0.4, 0.4, 1, 1], dtype=np.float64)
    # def axes_search_Ui_autarkie_diskont(i, delta_i, t_start, x, valid_constraints_only):
    #     Vi_autarkie, constraints = calc_Ui_autarkie_diskont(i, delta_i, t_start, x)
    #     if valid_constraints_only:
    #         return Vi_autarkie if constraints.min() >= min_constraints else np.nan
    #     else:
    #         return Vi_autarkie
    # axes_search(opt_fun = lambda x: axes_search_Ui_autarkie_diskont(1, delta_1, 0, x, True),  x0 = x0_autarkie, xmin = xmin_autarkie, xmax = xmax_autarkie, steps = 100, do_plot=True, title=f"axes_search(U1_autarkie_diskont, delta1={delta_1}, constraints>=0 only)")
    # axes_search(opt_fun = lambda x: axes_search_Ui_autarkie_diskont(1, delta_1, 0, x, False), x0 = x0_autarkie, xmin = xmin_autarkie, xmax = xmax_autarkie, steps = 100, do_plot=True, title=f"axes_search(U1_autarkie_diskont, delta1={delta_1}, constraints ignored)")
    # axes_search(opt_fun = lambda x: axes_search_Ui_autarkie_diskont(2, delta_2, 0, x, True),  x0 = x0_autarkie, xmin = xmin_autarkie, xmax = xmax_autarkie, steps = 100, do_plot=True, title=f"axes_search(U2_autarkie_diskont, delta2={delta_2}, constraints>=0 only)")
    # axes_search(opt_fun = lambda x: axes_search_Ui_autarkie_diskont(2, delta_2, 0, x, False), x0 = x0_autarkie, xmin = xmin_autarkie, xmax = xmax_autarkie, steps = 100, do_plot=True, title=f"axes_search(U2_autarkie_diskont, delta2={delta_2}, constraints ignored)")
    # plt.show()
    if verbose > 0:
        print(f"\nsearch_Ui_autarkie_diskont(): maximize U{i}_autarkie_diskont mit δ{i}={delta_i}, x0_autarkie aus Liste:")
    for x0_autarkie in [ np.array([6, 6, 8, 0, 0, 1, 1], dtype=np.float64), np.array([8, 6, 8, 0, 0, 1, 1], dtype=np.float64), 
                         np.array([6, 6, 8, 1e-3, 1e-3, 1.0, 1.0], dtype=np.float64), np.array([5, 5, 9, 1e-3, 1e-3, 1.00, 1.00], dtype=np.float64),
                         np.array([7, 7, 8, 1e-3, 1e-3, 1.0, 1.0], dtype=np.float64), np.array([7, 7, 8, 1e-3, 1e-3, 0.95, 0.95], dtype=np.float64), 
                         np.array([7, 7, 8, 0.01, 1e-3, 1.0, 1.0], dtype=np.float64), np.array([7, 7, 8, 1e-3, 0.01, 1.00, 1.00], dtype=np.float64),
                         np.array([6, 6, 8, 0.1, 0.1, 1.00, 1.00], dtype=np.float64), np.array([7, 7, 8, 0.1, 0.1, 1.00, 1.00], dtype=np.float64),
                         np.array([2, 1, 20, 0.1, 0.1, 0.5, 1.0], dtype=np.float64), np.array([2, 2, 18, 0.1, 0.1, 0.5, 1.0], dtype=np.float64),
                         np.array([4, 1, 15, 0.1, 0.1, 0.5, 1.0], dtype=np.float64), np.array([2, 1, 20, 0.1, 0.1, 0.5, 0.5], dtype=np.float64),
                         np.array([2, 1, 20, 0.1, 0.1, 0.3, 0.3], dtype=np.float64), np.array([2, 1, 20, 0.1, 0.1, 1.0, 0.5], dtype=np.float64) ]:
        x0_V, x0_constraints = calc_Ui_autarkie_diskont(i, delta_i, 0, x0_autarkie)
        if x0_constraints.min() < min_constraints:
            print(f"  search_Ui_autarkie_diskont(δ{i}={delta_i:.2f}): Startwerte x0_autarkie={fmt(x0_autarkie,'{:.3f}')} für Maximierung Autarkienutzen ungünstig, x0_autarkie verletzt Randbedingungen")
        sol_x, sol_U, sol_V, sol_constraints, sol_success = maximize_Ui_autarkie_diskont(i, delta_i, 0, x0_autarkie, method = "COBYLA", verbose=-1)
        if sol_success and verbose > 0:
            print(f"  search_Ui_autarkie_diskont(δ{i}={delta_i:.2f}): x0={fmt(x0_autarkie,'{:.3f}')}, x{i}={fmt(sol_x,'{:.3f}')}, U{i}={sol_U:.2f}, V{i}={sol_V:.4e}, min(constraints)={sol_constraints.min():.2e}")
    # Problem: Maximierung des diskontierten Autarkienutzens mit maximize_Ui_autarkie_diskont ist sensitiv gegenüber Startwerten x0_autarkie.
    # Daher Suche nach stabilem Startwert x0_autarkie durch schrittweise Erhöhung von delta_i:
    # * Start mit delta_i=0 (entspricht t_max=1), dadurch ist ein stabiler Startwert einfacher zu finden
    # * Danach schrittweise Erhöhung von delta_i mit Startwert aus vorherigem Ergebnis von maximize_Ui_autarkie_diskont, bis das gewünschte delta_i erreicht ist.
    delta_i_set = [0.00, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.00]
    if delta_i not in delta_i_set:
        delta_i_set.append(delta_i)
        delta_i_set.sort()
    xi_autarkie, Ui_autarkie, Vi_autarkie, constraintsi_autarkie = None, None, None, None
    # x0_autarkie = [ Li_fisch, Li_nuss, Li_subsistenz, lambdai_fisch_invest, lambdai_nuss_invest, wi_fisch, wi_nuss ]
    # Startwert delta_i=0: lambdai_fisch_invest=0, lambdai_nuss_invest=0, da sich Investitionen für Startwert delta_i=0 bzw. t_max=1 nicht auszahlen können.
    # Daher auch wi_fisch=1, wi_nuss=1 für delta_i=0 (Konsum ausschließlich für Subsistenz, da Investitionen nicht lohnen). 
    # Damit bleiben für delta_i=0 nur noch 3 Unbekannte: Li_fisch, Li_nuss, Li_subsistenz mit Li_subsistenz>=Li_subsistenz_min
    if verbose > 0:
        print(f"\nsearch_Ui_autarkie_diskont(): maximize U{i}_autarkie_diskont mit δ{i}=0, x0_autarkie aus Liste:")
    for x0_autarkie in [ np.array([6, 6, 8, 0, 0, 1, 1], dtype=np.float64), np.array([7, 6, 8, 0, 0, 1, 1], dtype=np.float64), np.array([7, 7, 8, 0, 0, 1, 1], dtype=np.float64), np.array([8, 6, 8, 0, 0, 1, 1], dtype=np.float64),
                         np.array([6, 6, 9, 0, 0, 1, 1], dtype=np.float64), np.array([7, 6, 9, 0, 0, 1, 1], dtype=np.float64), np.array([7, 7, 9, 0, 0, 1, 1], dtype=np.float64), np.array([8, 6, 9, 0, 0, 1, 1], dtype=np.float64),
                         np.array([6, 6, 10, 0, 0, 1, 1], dtype=np.float64), np.array([7, 6, 10, 0, 0, 1, 1], dtype=np.float64), np.array([6, 7, 10, 0, 0, 1, 1], dtype=np.float64) ]:
        x0_V, x0_constraints = calc_Ui_autarkie_diskont(i, 0, 0, x0_autarkie)
        if x0_constraints.min() < min_constraints:
            print(f"  search_Ui_autarkie_diskont(δ{i}=0): Startwerte x0_autarkie={fmt(x0_autarkie,'{:.3f}')} für Maximierung Autarkienutzen ungünstig, x0_autarkie verletzt Randbedingungen")
        sol_x, sol_U, sol_V, sol_constraints, sol_success = maximize_Ui_autarkie_diskont(i, 0, 0, x0_autarkie, method = "SLSQP", verbose=-1)
        if sol_success:
            if Vi_autarkie is None or sol_V > Vi_autarkie:
                xi_autarkie, Ui_autarkie, Vi_autarkie, constraintsi_autarkie = sol_x, sol_U, sol_V, sol_constraints
            if verbose > 0:
                print(f"  search_Ui_autarkie_diskont(δ{i}=0): x0={fmt(x0_autarkie,'{:.3f}')}, x{i}={fmt(sol_x,'{:.3f}')}, U{i}={sol_U:.2f}, V{i}={sol_V:.4e}, min(constraints)={sol_constraints.min():.2e}")
    if verbose > 0:
        print(f"  search_Ui_autarkie_diskont(δ{i}=0): x{i}_autarkie={fmt(xi_autarkie,'{:.3f}')}, U{i}_autarkie={Ui_autarkie:.2f}, V{i}_autarkie={Vi_autarkie:.4e}, min(constraints)={constraintsi_autarkie.min():.2e}")
    # Schrittweise Erhöhung von delta_i mit Startwert aus vorherigem Ergebnis von maximize_Ui_autarkie_diskont, bis das gewünschte delta_i erreicht ist.
    if verbose > 0:
        print(f"\nsearch_Ui_autarkie_diskont(): maximize U{i}_autarkie_diskont mit δ{i} ∈ {delta_i_set} und Startwerten aus vorherigem Optimum:")
    result_deltai = (xi_autarkie, Ui_autarkie, Vi_autarkie, constraintsi_autarkie)
    x0_autarkie = xi_autarkie
    for delta_i_set_order in range(2): # Stabilitätstest mit range(2) oder range(3): Mehrfacher Vorwärts- und Rückwärtsdurchlauf mit Startwert aus vorherigem Ergebnis
        for delta_i_step in delta_i_set:
            if delta_i_step > delta_i:
                break
            x0_V, x0_constraints = calc_Ui_autarkie_diskont(i, delta_i_step, 0, x0_autarkie)
            if x0_constraints.min() < min_constraints:
                print(f"  search_Ui_autarkie_diskont(δ{i}={delta_i_step:.2f}): Startwerte x0_autarkie={fmt(x0_autarkie,'{:.3f}')} für Maximierung Autarkienutzen ungünstig, x0_autarkie verletzt Randbedingungen")
            xi_autarkie, Ui_autarkie, Vi_autarkie, constraintsi_autarkie, sol_success = maximize_Ui_autarkie_diskont(i, delta_i_step, 0, x0_autarkie, method = "COBYLA", verbose=-1)
            if not sol_success:
                xi_autarkie, Ui_autarkie, Vi_autarkie, constraintsi_autarkie, sol_success = maximize_Ui_autarkie_diskont(i, delta_i_step, 0, x0_autarkie, method = "SLSQP", verbose=-1)
            if verbose > 0:
                print(f"  search_Ui_autarkie_diskont(δ{i}={delta_i_step:.2f}): x0={fmt(x0_autarkie,'{:.3f}')}, x{i}={fmt(xi_autarkie,'{:.3f}')}, U{i}={Ui_autarkie:.2f}, V{i}={Vi_autarkie:.4e}, min(constraints)={constraintsi_autarkie.min():.2e}")
            if np.isfinite(Ui_autarkie) and Vi_autarkie >= 0 and constraintsi_autarkie.min() >= min_constraints:
                x0_autarkie = xi_autarkie
            if delta_i_step == delta_i:
                result_deltai = (xi_autarkie, Ui_autarkie, Vi_autarkie, constraintsi_autarkie)
        delta_i_set.reverse() # Stabilitätstest: Mehrfacher Vorwärts- und Rückwärtsdurchlauf mit Startwert aus vorherigem Ergebnis
    xi_autarkie, Ui_autarkie, Vi_autarkie, constraintsi_autarkie = result_deltai
    if verbose > 0:
        print(f"  search_Ui_autarkie_diskont(δ{i}={delta_i:.2f}): x{i}={fmt(xi_autarkie,'{:.3f}')}, U{i}={Ui_autarkie:.2f}, V{i}={Vi_autarkie:.4e}, min(constraints)={constraintsi_autarkie.min():.2e}")
    return xi_autarkie, Ui_autarkie, Vi_autarkie, constraintsi_autarkie

# Maximierung der diskontierten Autarkienutzen inkl. Caching (Beschleunigung durch Vermeidung doppelter Berechnung der Autarkienutzen)
def maximize_Ui_autarkie_diskont_cached(i: Literal[1, 2], delta_i: float, t_start: float, x0: np.ndarray, autarkie_cache, verbose: int = 0):
    from mini_wm_robinson_two_persons import calc_Ui_autarkie_diskont, fmt, maximize_Ui_autarkie_diskont, min_constraints
    if delta_i in autarkie_cache:
        x_autarkie, U_autarkie, V_autarkie, constraints_autarkie, sol_success = autarkie_cache[delta_i]
    else:
        x_autarkie, U_autarkie, V_autarkie, constraints_autarkie, sol_success = maximize_Ui_autarkie_diskont(i, delta_i, t_start, x0, method = "COBYLA", verbose = verbose)
        if not sol_success:
            x_autarkie, U_autarkie, V_autarkie, constraints_autarkie, sol_success = maximize_Ui_autarkie_diskont(i, delta_i, t_start, x0, method = "SLSQP", verbose = verbose)
        if not sol_success or constraints_autarkie.min() < min_constraints:
            print(f"  ## maximize_nash_product(δ{i}={delta_i:.2f}): Maximierung der diskontierten Autarkienutzen fehlgeschlagen, success={sol_success}, min(constraints)={constraints_autarkie.min():.2e}")
        autarkie_cache[delta_i] = (x_autarkie, U_autarkie, V_autarkie, constraints_autarkie, sol_success)
    return x_autarkie, U_autarkie, V_autarkie, constraints_autarkie, sol_success

# Test der Stabilität von maximize_nash_product_x0 gegen Startwerte
def test_maximize_nash_product(delta_1, delta_2, x1_autarkie, x2_autarkie, V1_autarkie, V2_autarkie):
    from mini_wm_robinson_two_persons import calc_Ui_autarkie_diskont, fmt, maximize_nash_product_x0, maximize_Ui_autarkie_diskont, min_constraints

    # Test Stabilität gegen Startwerte
    print(f"\n  Stabilitätstest maximize_nash_product gegen Startwerte:")
    for np_x0 in [np.concatenate((x1_autarkie, [1e-3], x2_autarkie, [1e-3])), np.concatenate((x1_autarkie, [0.1], x2_autarkie, [0.1])), np.concatenate((x1_autarkie, [0.2], x2_autarkie, [0.2]))]:
        x, U1, U2, logNP, constraint, success = maximize_nash_product_x0(delta_1, delta_2, np_x0, V1_autarkie, V2_autarkie, True, verbose=1)

    # Test Stabilität mit Vorwärts- und Rückwärtsdurchlauf über delta_i ∈ delta_i_set
    print(f"\n  Stabilitätstest maximize_nash_product gegen Startwerte, Vorwärts- und Rückwärtsdurchlauf über delta_i ∈ delta_i_set:")
    delta_i_set = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
    delta_i_set.reverse()
    autarkie1_cache, autarkie2_cache = {}, {} # Caching (V1_autarkie, V2_autarkie) für jedes delta_i aus delta_i_set (Beschleunigung durch Vermeidung doppelter Berechnung der Autarkienutzen)
    # np_x0 = np.copy(x0)
    for delta_i_set_order in range(2): 
        for delta_i_step in delta_i_set:
            x1_autarkie, U1_autarkie, V1_autarkie, constraints1_autarkie, sol1_success = maximize_Ui_autarkie_diskont_cached(1, delta_i_step, 0, x1_autarkie, autarkie1_cache, verbose=-1)
            x2_autarkie, U2_autarkie, V2_autarkie, constraints2_autarkie, sol2_success = maximize_Ui_autarkie_diskont_cached(2, delta_i_step, 0, x2_autarkie, autarkie2_cache, verbose=-1)
            x0 = np.concatenate((x1_autarkie, [0.1], x2_autarkie, [0.1]))
            x, U1, U2, logNP, constraint, success = maximize_nash_product_x0(delta_i_step, delta_i_step, x0, V1_autarkie, V2_autarkie, False, verbose=1)
            # x, U1, U2, logNP, constraint, success = maximize_nash_product_x0(delta_i_step, delta_i_step, np_x0, V1_autarkie, V2_autarkie, False, verbose=0)
            # if success:
            #     np_x0 = x
            # best_x, best_U1, best_U2, best_logNP, best_constraint, best_success = maximize_nash_product_x0(delta_i_step, delta_i_step, np_x0, V1_autarkie, V2_autarkie, False, verbose=0)
            # print(f"    δ1=δ2={delta_i_step:.2f}, x0={fmt(np_x0,'{:.2f}')}: x={fmt(best_x,'{:.2f}')}, log(NP)={best_logNP:.2f}")
            # for n, x0 in enumerate([ np.concatenate((np_x0[0:7], [0.01], np_x0[8:-1], [0.01])), np.concatenate((np_x0[0:7], [0.1], np_x0[8:-1], [0.1])),
            #     np.concatenate((x1_autarkie, [0.01], x2_autarkie, [0.01])), np.concatenate((x1_autarkie, [0.1], x2_autarkie, [0.1])) ]):
            #     x, U1, U2, logNP, constraint, success = maximize_nash_product_x0(delta_i_step, delta_i_step, x0, V1_autarkie, V2_autarkie, False, verbose=0)
            #     if success and logNP > best_logNP:
            #         best_x, best_U1, best_U2, best_logNP, best_constraint, best_success = x, U1, U2, logNP, constraint, success
            #         print(f"    {n+1}, δ1=δ2={delta_i_step:.2f}, x0={fmt(x0,'{:.2f}')}: x={fmt(best_x,'{:.2f}')}, log(NP)={best_logNP:.2f}")
            # print(f"  δ1=δ2={delta_i_step:.2f}, x0={fmt(np_x0,'{:.2f}')}, x={fmt(best_x,'{:.2f}')}, U1={best_U1:.2f}, U2={best_U2:.2f}, log(NP)={best_logNP:.2f}, min(constraints)={best_constraint:.2e}")
            # if best_success:
            #     np_x0 = best_x
        delta_i_set.reverse()
    print()
