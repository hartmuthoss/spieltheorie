"""
Mini-WM: Ein minimalistisches Wirtschaftsmodell,
Teil 2: Robinson-Crusoe-Wirtschaft mit einem öffentlichem Gut.
Erläuterungen siehe MiniWM_Teil02_RobinsonFreitag.md.
"""

import datetime
import itertools
import math
import matplotlib.pyplot as plt
import numba
import numpy as np
import pathlib
import scipy # pip install scipy
import scipy.optimize
import sympy as sp # pip install sympy
import time

# numba.config.DISABLE_JIT = True # Disable numba.njit for debugging (Beschleunigung durch numba JIT-Compiler um ca. Faktor 2)
plt_cnt = 0 # Globaler Zähler für Diagramme und png-Dateien
plt_now_str = f"{datetime.datetime.now():%Y%m%d_%H%M%S}" # Globaler Zeitstempel für Diagramme und png-Dateien

# Shortcut zur Formatierung aller Elemente einer Liste
def fmt(list, fmt_str = "{:5.2f}", sep = ",", pre = "[", post="]"):
    return pre + sep.join([fmt_str.format(_elem) for _elem in list]) + post

# Speichert den aktuellen plot als png file
def save_plot_png(pngfile):
    if pngfile != "":
        pathlib.Path(pngfile).unlink(missing_ok=True)
        plt.savefig(pngfile)
        print(f"  Plot gespeichert in {pngfile}")

# Liefert den Namen einer Variablen als String, z.B. var_name(L1_holz) -> "L1_holz"
# namespace kann z.B. locals() oder globals() sein, default: locals()
def var_name(var, namespace = locals()):
    var_names = [name for name in namespace if namespace[name] is var]
    return "" if len(var_names) == 0 else var_names[0] if len(var_names) == 1 else var_names

# Plot von Kennwerten (debugging)
def rcw_02_plot_dbg(plots=[], xlabel="", ylabel="", block=True, figsize=(15,5)):
    plt.figure(figsize=figsize)
    for plot in plots:
        plt.plot(plot[0], plot[1], label = plot[2] if len(plot)>2 else var_name(plot[1]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show(block=block)
    if not block:
        plt.pause(1)

# Plot von Kennwerten über der Zeit
def rcw_02_plot(title, values):
    global plt_cnt
    plt.figure(figsize=(15,7)) 
    plt.title(title)
    # plt_fig, _ = plt.subplots(1, 2, figsize=(15,5))
    # plt_fig.suptitle(title)
    # plt.subplot(1, 2, 1)
    plt.plot(values.t, np.array(values.L2_schaden)/L_max, label=f"L2_schaden(t)/L_max")
    plt.plot(values.t, np.array(values.K_wald)/K_wald_max, label=f"K_wald(t)/K_wald_max")
    plt.plot(values.t, values.tau, label=f"tau(t)")
    plt.plot(values.t, values.q1, label=f"q(t)")
    # plt.plot(values.t, values.tau2_alternativ, label=f"tau2_alternativ(t)")
    plt.xlabel("Zeit t")
    plt.ylabel("K_wald, q, τ, Schaden")
    plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(values.t, np.array(values.L2_schaden), label=f"L2_schaden(t)")
    # plt.plot(values.t, np.array(values.q1)*L_max, label=f"L_max*q(t)")
    # plt.plot(values.t, values.U1_t, label=f"U1(t)")
    # plt.plot(values.t, values.U2_t, label=f"U2(t)")
    # plt.xlabel("Zeit t")
    # plt.ylabel("Nutzen U")
    # plt.legend()
    plt.show(block=False)
    plt_cnt += 1
    save_plot_png(f"mini_wm_rcw_02_race_to_preempt_{plt_now_str}_{plt_cnt:02d}.png")
    # plt.figure(figsize=(15,7))
    # plt.plot(values.t, values.U1_autarkie, label=f"U1_autarkie")
    # plt.plot(values.t, values.U2_autarkie, label=f"U2_autarkie")
    # plt.plot(values.t, values.U1_t, label=f"U1_t")
    # plt.plot(values.t, values.U2_t, label=f"U2_t")
    # plt.xlabel("Zeit t")
    # plt.ylabel("Nutzen U")
    # plt.legend()
    # plt.title(title)
    # plt.show(block=False)
    # plt_cnt += 1
    # save_plot_png(f"mini_wm_rcw_02_race_to_preempt_{plt_now_str}_{plt_cnt:02d}.png")
    plt.pause(1)

# grid_search: Maximierung von func(x,func_arg) mittels Suche über alle Kombinationen in einem Gitter mit gegebener Schrittweite
def rcw_grid_search(func, func_arg, x_start, x_stop, x_steps, U_min, constraints_fun = None, verbose=0):
    best_x = np.array(x_start, dtype=np.float32)
    best_y = U_min
    x_ranges = [np.arange(start, stop, step) for start, stop, step in zip(x_start, x_stop, x_steps)]
    for x in itertools.product(*x_ranges):
        if constraints_fun is not None and not constraints_fun(x):
            continue
        y = func(x, func_arg)
        if not np.isnan(y) and y > best_y:
            best_x = np.array(x, dtype=np.float32)
            best_y = y
            if verbose > 1:
                print("    grid_search: x={}, f(x)={}".format(x, y))
    return best_x, best_y

# grid_search: Maximierung von func(x) mittels Suche über alle Kombinationen in einem Gitter mit Iteration über abnehmende Schrittweiten
def rcw_grid_search_iter(func, func_arg, grid_x_start, grid_x_stop, grid_x_steps, num_iter=3, constraints_fun = None, verbose=0):
    assert(len(grid_x_start) == len(grid_x_stop) == len(grid_x_steps))
    sol_grid_x = grid_x_start
    sol_grid_U = 0
    for grid_iter in range(num_iter):
        grid_x, grid_U = rcw_grid_search(func, func_arg, x_start=grid_x_start, x_stop=grid_x_stop, x_steps=grid_x_steps, U_min = sol_grid_U, constraints_fun = constraints_fun, verbose = verbose)
        if grid_U > sol_grid_U:
            sol_grid_x = grid_x
            sol_grid_U = grid_U
        if verbose > 0:
            print("    grid_search: x={}, f(x)={:.2f}".format(sol_grid_x, sol_grid_U))
        for i in range(len(grid_x_start)):
            grid_x_start[i] = max(sol_grid_x[i] - grid_x_steps[i], grid_x_start[i])
            grid_x_stop[i]  = min(sol_grid_x[i] + grid_x_steps[i], grid_x_stop[i])
            grid_x_steps[i] *= 0.5
    return sol_grid_x, sol_grid_U

# numba-Wrapper-Klasse für globale Variablen, die in numba.njit-Funktionen verwendet werden
class ArrayWrapper:
    def __init__(self):
        self.set([])
    def set(self, params):
        self.params = [ x for x in params ]
    def get(self):
        return self.params

# Globale Variablen für die Übergabe von Parametern an numba.njit-Funktionen
global_stack = [ ArrayWrapper(), ArrayWrapper() ]

# numba-Wrapper-Klasse für Li, Yi, Ci, Ui, K, etc. über t für plots
@numba.experimental.jitclass
class RCWPlotValues:
    t: numba.types.ListType(numba.types.float64)
    L1_holz: numba.types.ListType(numba.types.float64)
    L1_alt: numba.types.ListType(numba.types.float64)
    L2_holz: numba.types.ListType(numba.types.float64)
    L2_schaden: numba.types.ListType(numba.types.float64)
    L2_alt: numba.types.ListType(numba.types.float64)
    Y_holz: numba.types.ListType(numba.types.float64)
    K_wald: numba.types.ListType(numba.types.float64)
    tau2_alternativ: numba.types.ListType(numba.types.float64)
    tau: numba.types.ListType(numba.types.float64)
    q1: numba.types.ListType(numba.types.float64)
    C1_alternativ: numba.types.ListType(numba.types.float64)
    C2_alternativ: numba.types.ListType(numba.types.float64)
    U1_autarkie: numba.types.ListType(numba.types.float64)
    U2_autarkie: numba.types.ListType(numba.types.float64)
    U1_t: numba.types.ListType(numba.types.float64)
    U2_t: numba.types.ListType(numba.types.float64)
    U1_diskont: numba.types.ListType(numba.types.float64)
    U2_diskont: numba.types.ListType(numba.types.float64)
    def __init__(self):
        self.t = numba.typed.List.empty_list(numba.types.float64)
        self.L1_holz = numba.typed.List.empty_list(numba.types.float64)
        self.L1_alt = numba.typed.List.empty_list(numba.types.float64)
        self.L2_holz = numba.typed.List.empty_list(numba.types.float64)
        self.L2_schaden = numba.typed.List.empty_list(numba.types.float64)
        self.L2_alt = numba.typed.List.empty_list(numba.types.float64)
        self.Y_holz = numba.typed.List.empty_list(numba.types.float64)
        self.K_wald = numba.typed.List.empty_list(numba.types.float64)
        self.tau2_alternativ = numba.typed.List.empty_list(numba.types.float64)
        self.tau = numba.typed.List.empty_list(numba.types.float64)
        self.q1 = numba.typed.List.empty_list(numba.types.float64)
        self.C1_alternativ = numba.typed.List.empty_list(numba.types.float64)
        self.C2_alternativ = numba.typed.List.empty_list(numba.types.float64)
        self.U1_autarkie = numba.typed.List.empty_list(numba.types.float64)
        self.U2_autarkie = numba.typed.List.empty_list(numba.types.float64)
        self.U1_t = numba.typed.List.empty_list(numba.types.float64)
        self.U2_t = numba.typed.List.empty_list(numba.types.float64)
        self.U1_diskont = numba.typed.List.empty_list(numba.types.float64)
        self.U2_diskont = numba.typed.List.empty_list(numba.types.float64)
    def append(self, t, L1_holz, L1_alt, L2_holz, L2_schaden, L2_alt, Y_holz, K_wald, tau2_alternativ, tau, q1, C1_alternativ, C2_alternativ, U1_autarkie, U2_autarkie, U1_t, U2_t, U1_diskont, U2_diskont):
        self.t.append(t)
        self.L1_holz.append(L1_holz)
        self.L1_alt.append(L1_alt)
        self.L2_holz.append(L2_holz)
        self.L2_schaden.append(L2_schaden)
        self.L2_alt.append(L2_alt)
        self.Y_holz.append(Y_holz)
        self.K_wald.append(K_wald)
        self.tau2_alternativ.append(tau2_alternativ)
        self.tau.append(tau)
        self.q1.append(q1)
        self.C1_alternativ.append(C1_alternativ)
        self.C2_alternativ.append(C2_alternativ)
        self.U1_autarkie.append(U1_autarkie)
        self.U2_autarkie.append(U2_autarkie)
        self.U1_t.append(U1_t)
        self.U2_t.append(U2_t)
        self.U1_diskont.append(U1_diskont)
        self.U2_diskont.append(U2_diskont)

# Settings für den race to preempt
@numba.experimental.jitclass
class RCWSettings:
    delta1: float
    delta2: float
    t_max:  int
    L2_schadenintensity: float
    def __init__(self, delta1, delta2, t_max, L2_schadenintensity):
        self.delta1 = delta1 # Robinsons Diskonktfaktor
        self.delta2 = delta2 # Freitags Diskonktfaktor
        self.t_max = t_max   # Maximale Anzahl von Zeiteinheiten, um die intertemporale Optimierung durchzuführen
        self.L2_schadenintensity = L2_schadenintensity # Schadenintensität, wenn Freitag auf Robinsons Tauschverhalten mit Abholzung zum Schaden aller reagiert

# Beispielparameter:
A1_holz = 2               # Robinsons Faktorproduktivität in der Holzproduktion
A2_holz = 0.5             # Freitags Faktorproduktivität in der Holzproduktion
A2_schaden = 1            # Freitags Faktorproduktivität in der Abholzung zum Schaden aller (Abholzung ohne Brennholzproduktion)
alpha1_holz = 0.5         # Robinsons Elastizität der Holzproduktion, Y1_holz = A1_holz * (L1_holz^alpha1_holz)
alpha2_holz = 0.5         # Freitags Elastizität der Holzproduktion, Y2_holz = A2_holz * (L2_holz^alpha2_holz)
alpha2_schaden = 0.5      # Freitags Elastizität der Abholzung zum Schaden aller, Y2_schaden = A2_schaden * (L2_schaden^alpha2_schaden)
A1_alternativ = 1         # Robinsons Faktorproduktivität in der Alternativproduktion
A2_alternativ = 1         # Freitags Faktorproduktivität in der Alternativproduktion
A_warme = 4               # Nutzenfaktor für die Wärme des gemeinsamen Lagerfeuers: Ui_holz = A_warme * (C1_holz + C2_holz)
L_max = 2                 # Maximal 2 Stunden Arbeit in der Holz- oder Alternativproduktion
K_wald_max = 100          # Maximale Größe des Waldes, z.B. 100 Bäume
g_wald = 0.1              # Natürliche Wachstumsrate des Waldes, z.B. 10% pro Zeiteinheit
# Robinsons Einschätzung q(t) wie glaubwürdig Freitags Drohung ist (erwartete Schadensintensität, d.h. Robinsons Einschätzung, welchen Anteil seiner maximalen Schadenskapazität Freitag im Konfliktfall einsetzen würde):  
g_q = 0.5                 # q(t+1) = g_q * q(t) mit konstantem Parameter 0 < g_q <= 1 falls L2_schaden(t)==0  
eta_q = 1.0               # q(t+1) = q(t) + eta_q * (L2_schaden(t) / L_max - q(t)) mit konstanter Anpassung 0 < eta_q <= 1 falls L2_schaden(t) > 0  

# Produktionsfunktionen
Y1_holz = numba.njit(lambda L1_holz: A1_holz * L1_holz**alpha1_holz)                      # Robinsons Brennholzproduktion für Lagerfeuer 
Y2_holz = numba.njit(lambda L2_holz: A2_holz * L2_holz**alpha2_holz)                      # Freitags  Brennholzproduktion für Lagerfeuer  
Y2_schaden = numba.njit(lambda L2_schaden: A2_schaden * L2_schaden**alpha2_schaden)       # Freitags  Abholzung zum Schaden aller
Y1_alternativ = numba.njit(lambda L1_alternativ: A1_alternativ * L1_alternativ)           # Robinsons Alternativproduktion  
Y2_alternativ = numba.njit(lambda L2_alternativ: A2_alternativ * L2_alternativ)           # Freitags  Alternativproduktion  

# Zeitrestriktion:
L1_alternativ = numba.njit(lambda L1_holz: L_max - L1_holz)                               # L1_holz + L1_alternativ = L_max
L2_alternativ = numba.njit(lambda L2_holz, L2_schaden: L_max - L2_holz - L2_schaden)      # L2_holz + L2_schaden + L2_alternativ = L_max

# Nutzenfunktionen
U1_holz_autarkie = numba.njit(lambda C1_holz: A_warme * C1_holz)                          # Robinsons Nutzen aus Holzkonsum unter Autarkie
U2_holz_autarkie = numba.njit(lambda C2_holz: A_warme * C2_holz)                          # Freitags  Nutzen aus Holzkonsum unter Autarkie
Ui_holz_kooperation = numba.njit(lambda C1_holz, C2_holz: A_warme * (C1_holz + C2_holz))  # Nutzen aus Holzkonsum bei Kooperation (gemeinsames Lagerfeuer)
U1_alternativ = numba.njit(lambda C1_alternativ: C1_alternativ)                           # Robinsons Nutzen aus Alternativkonsum
U2_alternativ = numba.njit(lambda C2_alternativ: C2_alternativ)                           # Freitags  Nutzen aus Alternativkonsum

# Autarkie-Nutzen
U1_autarkie = numba.njit(lambda L1_holz_autarkie: U1_holz_autarkie(Y1_holz(L1_holz_autarkie)) + U1_alternativ(Y1_alternativ(L1_alternativ(L1_holz_autarkie))))
U2_autarkie = numba.njit(lambda L2_holz_autarkie, L2_schaden_autarkie: U2_holz_autarkie(Y2_holz(L2_holz_autarkie)) + U2_alternativ(Y2_alternativ(L2_alternativ(L2_holz_autarkie, L2_schaden_autarkie))))

# Der Wald wird als begrenztes Kapital modelliert
@numba.njit 
def calc_K_wald_update(K_wald, L1_holz, L2_holz, L2_schaden):
    if K_wald <= 0: # Kein Wald => keine Abholzung
        return 0, 0, 0, 0, 0
    Y_holz = Y1_holz(L1_holz) + Y2_holz(L2_holz) + Y2_schaden(L2_schaden) # Gesamte Abholzung
    K_wald_left = K_wald - Y_holz # Waldbestand nach Abholzung mit Y_holz
    if K_wald_left >= 0:
        K_wald = K_wald_left + (g_wald * K_wald_left) * (1 - K_wald_left / K_wald_max)
    else: # Ressourcenrestriktion: Robinson und Freitag können nicht mehr Holz fällen als Bäume existieren:
        if K_wald > 0 and Y_holz > 0: 
            # Aufteilung des Restwaldes:
            # Produktionsfunktionen: Y1_holz = A1_holz*L1_holz^alpha1_holz, Y2_holz = A2_holz*L2_holz^alpha2_holz, Y2_schaden = A2_schaden*L2_schaden^alpha2_schaden
            # Ressourcenrestriktion: Y1_holz(L1_holz_rest) + Y2_holz(L2_holz_rest) + Y2_schaden(L2_schaden_rest) = K_wald
            # Skalierung der Restproduktion: Y1_holz(L1_holz_rest) = Y1_holz(L1_holz)*K_wald/Y_holz, Y2_holz(L2_holz_rest) = Y2_holz(L2_holz)*K_wald/Y_holz, Y2_schaden(L2_schaden_rest) = Y2_schaden(L2_schaden)*K_wald/Y_holz
            # => A1_holz*L1_holz_rest^alpha1_holz = (A1_holz*L1_holz^alpha1_holz)*K_wald/Y_holz, A2_holz*L2_holz_rest^alpha2_holz = (A2_holz*L2_holz^alpha2_holz)*K_wald/Y_holz, A2_schaden*L2_schaden_rest^alpha2_schaden = (A2_schaden*L2_schaden^alpha2_schaden)*K_wald/Y_holz
            # => L1_holz_rest^alpha1_holz = L1_holz^alpha1_holz*K_wald/Y_holz, L2_holz_rest^alpha2_holz = L2_holz^alpha2_holz*K_wald/Y_holz, L2_schaden_rest^alpha2_schaden = L2_schaden^alpha2_schaden*K_wald/Y_holz
            # => L1_holz_rest = (L1_holz^alpha1_holz*K_wald/Y_holz)^(1/alpha1_holz), L2_holz_rest = (L2_holz^alpha2_holz*K_wald/Y_holz)^(1/alpha2_holz), L2_schaden_rest = (L2_schaden^alpha2_schaden*K_wald/Y_holz)^(1/alpha2_schaden)
            L1_holz = (L1_holz**alpha1_holz * K_wald / Y_holz)**(1/alpha1_holz)
            L2_holz = (L2_holz**alpha2_holz * K_wald / Y_holz)**(1/alpha2_holz)
            L2_schaden = (L2_schaden**alpha2_schaden * K_wald / Y_holz)**(1/alpha2_schaden)
        else: # Ohne Wald keine Holzproduktion
            L1_holz, L2_holz, L2_schaden = 0, 0, 0
        K_wald = 0
        Y_holz = Y1_holz(L1_holz) + Y2_holz(L2_holz) + Y2_schaden(L2_schaden) # Restliche Abholzung
    return K_wald, L1_holz, L2_holz, L2_schaden, Y_holz

# Berechnet die Zeit bis der Wald komplett ausgebeutet ist, d.h. t mit K_wald(t) <= 0, wenn in jedem Zeitschritt t_start <= t < t_max die Abholzung Y_holz erfolgt.
# Ist der Wald bis t_max nicht ausgebeutet, wird t_max zurückgegeben.
@numba.njit 
def calc_time_when_exploited(t_start, t_max, K_wald_start, Y_holz):
    # Man kann die Berechnung der Zeit bis zur Ausbeutung des Waldes evtl. approximieren und dann direkt lösen. Ansatz:
    # K(t+1) = K(t) - Y + g * (K(t) - Y) * (1 - (K(t) - Y) / K_wald_max) mit K(t=t_start) = K_wald_start, Y = Y_holz
    # dK/dt ≈ ΔK/Δt = (K(t+1)-K(t))/(t+1-t) = K(t+1) - K(t) = -Y + g * (K(t) - Y) * (1 - (K(t) - Y) / K_wald_max)
    # K(t) ≈ Integral von dK/dt + C mit K(t_start) = K_wald_start
    K_wald = K_wald_start
    for t in range(t_start, t_max):
        K_wald_left = K_wald - Y_holz
        if K_wald_left <= 0:
            return t + K_wald/Y_holz
        K_wald = K_wald_left + (g_wald * K_wald_left) * (1 - K_wald_left / K_wald_max)  
    return t_max

# Berechnung der diskontierten Autarkie-Nutzens U1 zum Zeitpunkt t:
@numba.njit 
def calc_U1_autarkie_diskont(x, t_start, t_max, K_wald_start, delta1, q1, L2_holz_max, L2_schaden_max):
    # Waldentwicklung unter Autarkie: Robinson nimmt eine Abholzungsaktivität durch Freitag entsprechend q1 an, Y2_holz = A2_schaden*(q1*L2_schaden)^alpha2_schaden + A2_holz*((1-q1)*L2_holz)^alpha2_holz
    L1_holz = x[0]
    U1_autarkie_val0 = U1_autarkie(L1_holz)
    U1_autarkie_val1 = U1_autarkie(0)
    t_exploited = calc_time_when_exploited(t_start, t_max, K_wald_start, Y2_schaden(q1 * L2_schaden_max) + Y2_holz((1 - q1) * L2_holz_max) + Y1_holz(L1_holz))
    # Ui_autarkie_diskont(t) = sum_{n=t}^{t_max} delta_i^{n-t} Ui_autarkie(n) -> max
    if delta1 == 1:
        U1_diskont = max(0, t_exploited - t_start) * U1_autarkie_val0 + max(0, t_max - t_exploited) * U1_autarkie_val1
    else:
        U1_diskont = 0
        for t in range(t_start, t_exploited):
            U1_diskont += ((delta1**(t-t_start)) * U1_autarkie_val0)
        for t in range(t_exploited, t_max):
            U1_diskont += ((delta1**(t-t_start)) * U1_autarkie_val1)
    return U1_diskont, t_exploited
    
# numba wrapper zum Aufruf von calc_U1_autarkie_diskont mit Übergabe der Parameter über globale Variablen
def calc_U1_autarkie_diskont_numba_wrapper(x):
    t, t_max, K_wald, delta1, q1, L2_holz_max, L2_schaden_max = global_stack[0].get()
    U1_diskont, t_wald_exploited = calc_U1_autarkie_diskont(x, t, t_max, K_wald, delta1, q1, L2_holz_max, L2_schaden_max)
    # print("  calc_U1_autarkie_diskont_numba_wrapper: x={}, t={:.2f}, K_wald={:.2f}, delta1={:.2f}, q1={:.2f}, L2_holz_max={:.2f}, L2_schaden_max={:.2f} => U1_diskont={:.2f}".format(x, t, K_wald, delta1, q1, L2_holz_max, L2_schaden_max, U1_diskont))
    return -U1_diskont

# Berechnung der diskontierten Autarkie-Nutzens U2 zum Zeitpunkt t:
@numba.njit 
def calc_U2_autarkie_diskont(x, t_start, t_max, K_wald_start, delta2, L1_holz_max):
    # Waldentwicklung unter Autarkie: Freitag nimmt maximale Abholzung durch Robinson an, Y1_holz = A1_holz * L1_holz^alpha1_holz
    L2_holz = x[0]
    U2_autarkie_val0 = U2_autarkie(L2_holz, 0)
    U2_autarkie_val1 = U2_autarkie(0, 0)
    t_exploited = calc_time_when_exploited(t_start, t_max, K_wald_start, Y1_holz(L1_holz_max) + Y2_holz(L2_holz))
    # Ui_autarkie_diskont(t) = sum_{n=t}^{t_max} delta_i^{n-t} Ui_autarkie(n) -> max
    if delta2 == 1:
        U2_diskont = max(0, t_exploited - t_start) * U2_autarkie_val0 + max(0, t_max - t_exploited) * U2_autarkie_val1
    else:
        U2_diskont = 0
        for t in range(t_start, t_exploited):
            U2_diskont += ((delta2**(t-t_start)) * U2_autarkie_val0)
        for t in range(t_exploited, t_max):
            U2_diskont += ((delta2**(t-t_start)) * U2_autarkie_val1)
    return U2_diskont, t_exploited

# numba wrapper zum Aufruf von calc_U2_autarkie_diskont mit Übergabe der Parameter über globale Variablen
def calc_U2_autarkie_diskont_numba_wrapper(x):
    t, t_max, K_wald, delta2, L1_holz_max = global_stack[1].get()
    U2_diskont, t_wald_exploited = calc_U2_autarkie_diskont(x, t, t_max, K_wald, delta2, L1_holz_max)
    # print("  calc_U2_autarkie_diskont_numba_wrapper: x={}, t={:.2f}, K_wald={:.2f}, delta2={:.2f}, L1_holz_max={:.2f} => U2_diskont={:.2f}".format(x, t, K_wald, delta2, L1_holz_max, U2_diskont))
    return -U2_diskont

# Berechnung der Autarkie-Nutzen unter Annahme maximaler Abholzung durch den jeweils anderen:
# Robinson nimmt maximale Abholzung durch Freitag mit L2_schaden=L_max an, 
# Freitag nimmt maximale Abholzung durch Robinson mit L1_holz=L_max an
# Die Autarkie-Nutzen Ui_autarkie werden für K_wald > 0 durch Maximierung der diskontierten Autarkie-Nutzen berechnet:
# Li_holz_autarkie(t) werden so bestimmt, dass der diskontierte Nutzen unter Autarkiebedingungen maximal wird:  
# Ui_autarkie_diskont(t) = sum_{n=t}^{t_max} delta_i^{n-t} Ui_autarkie(n) -> max  
# Zurückgegeben werden Ui_autarkie = Ui_holz(Yi_holz(Li_holz_autarkie)) + Ui_alternativ(Yi_alternativ(Li_alternativ_autarkie))
@numba.njit 
def calc_U_autarkie(t, t_max, K_wald, Y_holz, q1, delta1, delta2, L1_holz_max = L_max, L2_holz_max = L_max, L2_schaden_max = L_max, verbose = 0):
    if K_wald > 0: # Holzentnahme aus Wald
        K_wald_available = K_wald
    elif Y_holz > 0: # Restholz vor Kahlschlag
        K_wald_available = Y_holz
    else: # Kein Wald, kein Holz => Kein Tauschhandel, Nutzen ausschließlich aus Alternativproduktion
        K_wald_available = 0
    if K_wald_available <= 0: # Kein Wald, kein Holz => Kein Tauschhandel, Nutzen ausschließlich aus Alternativproduktion
        return U1_alternativ(Y1_alternativ(L_max)), U2_alternativ(Y2_alternativ(L_max))
    # U1_autarkie_diskont(t) = sum_{n=t}^{t_max} delta_i^{n-t} U1_autarkie(n) -> max, x[0] = L1_holz_autarkie
    # sol1 = scipy.optimize.minimize(fun = lambda x: -calc_U1_autarkie_diskont(x, t, K_wald_available, delta1, q1, min(L2_holz_max, L_max - L2_schaden_max), L2_schaden_max)[0], x0 = [L1_holz_max], bounds = ((0, L1_holz_max),), method = "SLSQP", options = {"maxiter": 1000, "ftol": 1e-6})
    if K_wald_available > Y2_schaden(q1 * L2_schaden_max) + Y2_holz((1 - q1) * L2_holz_max):
        with numba.objmode(L1_holz_autarkie='float64'):
            global_stack[0].set([t, t_max, K_wald_available, delta1, q1, L2_holz_max, L2_schaden_max])
            sol1 = scipy.optimize.minimize(fun = calc_U1_autarkie_diskont_numba_wrapper, x0 = [L1_holz_max], bounds = ((0, L1_holz_max),), method = "SLSQP", options = {"maxiter": 1000, "ftol": 1e-6})
            if sol1.success:
                L1_holz_autarkie = sol1.x[0]
                if verbose > 0:
                    print("  calc_U_autarkie(t={:.2f}, K_wald={:.2f}, q1={:.2f}, delta1={:.2f}): sol1.x={:.2f}, U1_autarkie_diskont={:.2f}".format(t, K_wald_available, q1, delta1, sol1.x[0], -sol1.fun))
            else:
                L1_holz_autarkie = L1_holz_max
                print("  calc_U_autarkie(t={:.2f}, K_wald={:.2f}, q1={:.2f}, delta1={:.2f}): scipy.optimize.minimize failed to maximize U1_autarkie (\"{}\")".format(t, K_wald_available, q1, delta1, sol1.message))
    else: 
        L1_holz_autarkie = 0 # Annahme: Der andere kann den Restwald zuerst abholzen. Alternative: Aufteilung des Restholzes, sofern K_wald_available > 0.
    # U2_autarkie_diskont(t) = sum_{n=t}^{t_max} delta_i^{n-t} U2_autarkie(n) -> max, x[0] = L2_holz_autarkie
    # sol2 = scipy.optimize.minimize(fun = lambda x: -calc_U2_autarkie_diskont(x, t, K_wald_available, delta2, L1_holz_max)[0], x0 = [L2_holz_max], bounds = ((0, L2_holz_max),), method = "SLSQP", options = {"maxiter": 1000, "ftol": 1e-6})
    if K_wald_available > Y1_holz(L1_holz_max):
        with numba.objmode(L2_holz_autarkie='float64'):
            global_stack[1].set([t, t_max, K_wald_available, delta2, L1_holz_max])
            sol2 = scipy.optimize.minimize(fun = calc_U2_autarkie_diskont_numba_wrapper, x0 = [L2_holz_max], bounds = ((0, L2_holz_max),), method = "SLSQP", options = {"maxiter": 1000, "ftol": 1e-6})
            if sol2.success:
                L2_holz_autarkie = sol2.x[0]
                if verbose > 0:
                    print("  calc_U_autarkie(t={:.2f}, K_wald={:.2f}, delta2={:.2f}): sol2.x={:.2f}, U2_autarkie_diskont={:.2f}".format(t, K_wald_available, delta2, sol2.x[0], -sol2.fun))
            else:
                L2_holz_autarkie = L2_holz_max
                print("  calc_U_autarkie(t={:.2f}, K_wald={:.2f}, delta2={:.2f}): scipy.optimize.minimize failed to maximize U2_autarkie (\"{}\")".format(t, K_wald_available, delta2, sol2.message))
    else:
        L2_holz_autarkie = 0 # Annahme: Der andere kann den Restwald zuerst abholzen. Alternative: Aufteilung des Restholzes, sofern K_wald_available > 0.
    return U1_autarkie(L1_holz_autarkie), U2_autarkie(L2_holz_autarkie, 0)

# Der Tauschanteil tau2_alternativ Wärme gegen Alternativprodukt wird so bestimmt, dass das Nashprodukt (U1(t) - U1_autarkie(t)) * (U2(t) - U2_autarkie(t)) maximal wird.
# Rückgabe: Tauschanteil tau2_alternativ und Tauschpreis tau Wärme gegen Alternativprodukt
@numba.njit 
def calc_tau2_alternativ(L1_holz, L2_holz, L2_schaden, U1_autarkie_t, U2_autarkie_t):
    Y1_alt = Y1_alternativ(L1_alternativ(L1_holz))
    Y2_alt = Y2_alternativ(L2_alternativ(L2_holz, L2_schaden))
    C1_holz = Y1_holz(L1_holz)
    C2_holz = Y2_holz(L2_holz)
    # Tauschanteil tau2_alternativ:
    if C1_holz + C2_holz <= 0: # Kein Brennholz => Kein Tauschhandel
        tau2_alternativ = 0.0
    elif Y2_alt <= 0:
        tau2_alternativ = 0.0 # Keine Alternativprodukte => Kein Tauschhandel
    elif C1_holz <= C2_holz:
        tau2_alternativ = 0.0 # Freitag tauscht nur, wenn er im Gegenzug Wärme dafür erhält. Ist Robinsons Holzanteil am gemeinsamen Lagerfeuer kleiner als Freitags Holzanteil, dann kein Tausch.
    else:
        # Der Tauschanteil tau2_alternativ Wärme gegen Alternativprodukt wird so bestimmt, dass das Nashprodukt (U1(t) - U1_autarkie(t)) * (U2(t) - U2_autarkie(t)) maximal wird.
        # nashprod(tau2_alternativ) = (U_warme + Y1_alt + tau2_alternativ * Y2_alt - U1_autarkie) * (U_warme + Y2_alt - tau2_alternativ * Y2_alt - U2_autarkie) -> max
        # Analytische Lösung der Parabelgleichung nashprod(tau2_alternativ): ∂_nashprod/∂_tau2_alternativ = 0
        # U_warme, U1_autarkie, U2_autarkie, Y1_alt, Y2_alt, tau2_alternativ = sp.symbols("U_warme U1_autarkie U2_autarkie Y1_alt Y2_alt tau2_alternativ", real=True, positive=True)
        # nashprod = (U_warme + Y1_alt + tau2_alternativ * Y2_alt - U1_autarkie) * (U_warme + Y2_alt - tau2_alternativ * Y2_alt - U2_autarkie)
        # d_nashprod_d_tau = sp.diff(nashprod, tau2_alternativ)
        # sol = sp.solve(sp.Eq(d_nashprod_d_tau, 0), tau2_alternativ) 
        # Ergebnis: sol[0] = tau2_alternativ = (U1_autarkie - U2_autarkie - Y1_alt + Y2_alt) / (2 * Y2_alt)
        tau2_alternativ = (U1_autarkie_t - U2_autarkie_t - Y1_alt + Y2_alt) / (2 * Y2_alt)
        tau2_alternativ = max(0.0, min(1.0, tau2_alternativ)) # Freitag kann nicht mehr liefern als er produzieren kann: 0 <= tau2_alternativ <= 1
    # Tauschpreis tau = von Freitag verlangte Tauschmenge geteilt durch Robinsons Mehranteil am Feuerholz
    tau = tau2_alternativ * Y2_alt / (C1_holz - C2_holz) if C1_holz > C2_holz else 0 
    return tau2_alternativ, tau

# Berechnung von Robinsons Einschätzung 0 <= q1(t) <= 1, wie glaubwürdig Freitags Drohung ist, d.h. welche Schadensintensität Robinson von Freitag erwartet:
@numba.njit 
def calc_q1(t, q1, L2_schaden, L2_schadenintensity):
    if L2_schaden > 0:
        q1 = q1 + eta_q * (L2_schaden / L2_schadenintensity - q1)
    elif t > 0:
        q1 = g_q * q1
    return q1

# Berechnung der diskontierten Nutzenfunktion U2_diskont(t) mit und ohne Schaden (L2_schaden_start = 0 oder L2_schaden_start = L_max)
# zum Vergleich, ob sich eine einmalige Waldschädigung in der nächsten Periode für Freitag lohnt, d.h. ob U2_diskont(t) mit Schaden > U2_diskont(t) ohne Schaden.
# Zurückgegeben wird der diskontierte Kooperationsüberschuss sum((delta2**(t - t_start)) * (U2(t) - U2_autarkie(t))).
@numba.njit 
def calc_U2_diskont_schaden(delta1, delta2, t_start, t_max, K_wald_start, L1_holz_start, L2_holz_start, L2_schaden_start, q1_start, L2_schadenintensity):
    U2_diskont = 0
    K_wald = K_wald_start
    q1 = q1_start
    L2_schaden = L2_schaden_start
    for t in range(t_start, t_max):
        # Waldentwicklung
        K_wald, L1_holz, L2_holz, L2_schaden, Y_holz = calc_K_wald_update(K_wald, L1_holz_start, min(L_max - L2_schaden, L2_holz_start), L2_schaden)
        # Robinsons Einschätzung 0 <= q1(t) <= 1, wie glaubwürdig Freitags Drohung ist (erwartete Schadensintensität)
        q1 = calc_q1(t - t_start, q1, L2_schaden, L2_schadenintensity)
        # Autarkie-Nutzen unter Annahme maximaler Abholzung durch den jeweils anderen (Robinson nimmt maximale Abholzung durch Freitag mit L2_schaden=L_max an, Freitag nimmt maximale Abholzung durch Robinson mit L1_holz=L_max an)
        U1_autarkie_t, U2_autarkie_t = calc_U_autarkie(t, t_max, K_wald, Y_holz, q1, delta1, delta2, L_max, L_max, L_max)
        # Tauschanteil tau2_alternativ
        tau2_alternativ, _ = calc_tau2_alternativ(L1_holz, L2_holz, L2_schaden, U1_autarkie_t, U2_autarkie_t)
        # Nutzen U2(t) bei Kooperation
        Ui_warme = Ui_holz_kooperation(Y1_holz(L1_holz), Y2_holz(L2_holz))
        C2_alternativ = (1 - tau2_alternativ) * Y2_alternativ(L2_alternativ(L2_holz, L2_schaden))
        U2_t = Ui_warme + U2_alternativ(C2_alternativ)
        # Diskontierter Nutzen bei Kooperation (gemeinsames Lagerfeuer und Tauschhandel):
        if U2_t < U2_autarkie_t:
            return 0.0 # Harte Entscheidung: Kooperation hat keinen Vorteil gegenüber Autarkie -> kein Tauschhandel, Kooperationsüberschuss = 0
        # Alternativ: Weiche Entscheidung, der Kooperationsüberschuss wird in dieser Periode ausgeschlossen, nicht aber in Zukunft. Praktisch kein Unterschied.
        # Freitag möchte den Preis drücken; das setzt Kooperation voraus. Freitag vergleicht daher nicht seinen absoluten Konsumnutzen, sondern den diskontierten Wert der Kooperation gegenüber Autarkie. 
        U2_diskont += (delta2**(t - t_start)) * max(0.0, U2_t - U2_autarkie_t) 
        # Einmaliger Schaden zum Vergleich der diskontierten Nutzenfunktion U2_diskont(t) mit und ohne Schaden => L2_schaden(t+1) = 0
        L2_schaden = 0
    return U2_diskont

# Ist der Preis zu hoch, reagiert Freitag auf Robinsons Tauschverhalten in der nächsten Periode mit Abholzung zum Schaden aller. 
# Alternative Reaktionsfunktionen: L2_schaden(t+1) = L_max (maximale Reakton), oder 
# L2_schaden(t+1) = L_max * (1 - e^(-β_tau*(tau-tau_thresh))) mit Eskalationshärte β_tau, d.h. je höher der Preis, desto stärker die Reaktion. Dann begrenzt Freitag den Schaden, kann aber mit Eskalation drohen.
# 2-Parameter-Modell mit 2 Unbekannte (L1_holz, L2_holz): Berechnung der diskontierten Nutzenfunktion U2_diskont(t) mit und ohne Schaden zum Vergleich, ob sich eine einmalige Waldschädigung für Freitag lohnt, 
# d.h. ob U2_diskont(t) mit Schaden > U2_diskont(t) ohne Schaden. Wenn ja, dann L2_schaden(t+1) = L_max oder L2_schaden(t+1) = L_max * (1 - e^(-β_tau*(tau-tau_thresh))), andernfalls L2_schaden(t+1) = 0.
@numba.njit 
def calc_L2_schaden(delta1, delta2, t, t_max, K_wald, L1_holz, L2_holz, q1, L2_schadenintensity, tau, verbose = 0):
    if K_wald > 0:
        # Berechnung der diskontierten Kooperationsüberschüsse mit und ohne Schaden (L2_schaden_start = 0 oder L2_schaden_start = L_max)
        # zum Vergleich, ob sich eine einmalige Waldschädigung in der nächsten Periode für Freitag lohnt, d.h. ob U2_diskont(t) mit Schaden > U2_diskont(t) ohne Schaden.
        # Freitags Schadensentscheidung vergleicht Kooperationsüberschüsse, nicht absoluten Nutzen.
        L2_schaden = L2_schadenintensity
        U2_diskont_0 = calc_U2_diskont_schaden(delta1, delta2, t, t_max, K_wald, L1_holz, L2_holz, 0, q1, L2_schadenintensity)
        U2_diskont_1 = calc_U2_diskont_schaden(delta1, delta2, t, t_max, K_wald, L1_holz, L2_holz, L2_schaden, q1, L2_schadenintensity)
        if U2_diskont_0 >= U2_diskont_1:
            if verbose > 0:
                print("  t=", t, ", K_wald=", K_wald, ", tau=", tau, ": U2_diskont_schaden(L2_schaden=0)=", U2_diskont_0, " >= U2_diskont_schaden(L2_schaden=", L2_schaden, ")=", U2_diskont_1, " => L2_schaden=0")
            L2_schaden = 0
        elif verbose > 0:
            print("  t=", t, ", K_wald=", K_wald, ", tau=", tau, ": U2_diskont_schaden(L2_schaden=0)=", U2_diskont_0, " < U2_diskont_schaden(L2_schaden=", L2_schaden, ")=", U2_diskont_1, " => L2_schaden=", L2_schaden)
    else:
        L2_schaden = 0
    return L2_schaden

# Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
# Nashprodukt = sum_{t=0}^{t=t_max} (delta_1^t * (U1(t) - U1_autarkie(t))) * sum_{t=0}^{t=t_max} (delta_2^t * (U2(t) - U2_autarkie(t)))
# 2 Unbekannte: x[0] = L1_holz, x[1] = L2_holz
@numba.njit 
def calc_nash_product(x, delta1, delta2, t_max, L2_schadenintensity, export_values = False):
    assert(len(x) == 2)
    U1_diskont, U2_diskont = 0, 0
    constraints_failure = 0
    K_wald = K_wald_max
    t_wald_exploited = t_max + 1
    q1 = 1
    L2_schaden = 0
    values = RCWPlotValues() if export_values else None
    for t in range(t_max):
        L1_holz, L2_holz = x[0], x[1]
        L2_holz = min(L2_holz, L_max - L2_schaden) # Zeitrestriktion: L2_holz + L2_schaden <= L_max
        # Der Wald wird als begrenztes Kapital modelliert
        K_wald, L1_holz, L2_holz, L2_schaden, Y_holz = calc_K_wald_update(K_wald, L1_holz, L2_holz, L2_schaden) # x[0] = L1_holz, x[1] = L2_holz
        assert(L1_holz == 0 and L2_holz == 0 and L2_schaden == 0 if Y_holz <= 0 else True)
        # Robinsons Einschätzung 0 <= q1(t) <= 1, wie glaubwürdig Freitags Drohung ist (erwartete Schadensintensität)
        q1 = calc_q1(t, q1, L2_schaden, L2_schadenintensity)
        # Autarkie-Nutzen unter Annahme maximaler Abholzung durch den jeweils anderen (Robinson nimmt maximale Abholzung durch Freitag mit L2_schaden=L_max an, Freitag nimmt maximale Abholzung durch Robinson mit L1_holz=L_max an)
        U1_autarkie_t, U2_autarkie_t = calc_U_autarkie(t, t_max, K_wald, Y_holz, q1, delta1, delta2, L_max, L_max, L_max)
        if K_wald <= 0 and Y_holz <= 0:
            t_wald_exploited = min(t, t_wald_exploited)
        # Tauschanteil tau2_alternativ und Tauschpreis tau
        tau2_alternativ, tau = calc_tau2_alternativ(L1_holz, L2_holz, L2_schaden, U1_autarkie_t, U2_autarkie_t)
        # Nutzen bei Kooperation (gemeinsames Lagerfeuer und Tauschhandel)
        L1_alt = L1_alternativ(L1_holz)
        L2_alt = L2_alternativ(L2_holz, L2_schaden)
        Y1_alt = Y1_alternativ(L1_alt)
        Y2_alt = Y2_alternativ(L2_alt)
        C1_holz = Y1_holz(L1_holz)
        C2_holz = Y2_holz(L2_holz)
        Ui_warme = Ui_holz_kooperation(C1_holz, C2_holz)
        C1_alternativ = Y1_alt + tau2_alternativ * Y2_alt
        C2_alternativ = Y2_alt - tau2_alternativ * Y2_alt
        U1_t = Ui_warme + U1_alternativ(C1_alternativ)
        U2_t = Ui_warme + U2_alternativ(C2_alternativ)
        # Diskontierter Nutzen bei Kooperation (gemeinsames Lagerfeuer und Tauschhandel) gegenüber Autarkie
        constraints_failure = min(constraints_failure, U1_t - U1_autarkie_t, U2_t - U2_autarkie_t) # constraints_failure < 0: Verletzung der Bedingungen U1_t >= U1_autarkie, U2_t >= U2_autarkie
        U1_diskont += (delta1**t) * (U1_t - U1_autarkie_t)
        U2_diskont += (delta2**t) * (U2_t - U2_autarkie_t)
        # Ausgabe Li, Ui, K über t für plots
        if export_values:
            values.append(t, L1_holz, L1_alt, L2_holz, L2_schaden, L2_alt, Y_holz, K_wald, tau2_alternativ, tau, q1, C1_alternativ, C2_alternativ, U1_autarkie_t, U2_autarkie_t, U1_t, U2_t, U1_diskont, U2_diskont)
        # Ist der Preis zu hoch, reagiert Freitag auf Robinsons Tauschverhalten mit Abholzung in der nächsten Periode, zum Schaden aller. 
        # Berechnung der diskontierten Nutzenfunktion U2_diskont(t) mit und ohne Schaden zum Vergleich, ob sich eine einmalige Waldschädigung für Freitag lohnt. Falls ja, dann L2_schaden(t+1) > 0, ansonsten L2_schaden(t+1) = 0.
        L2_schaden = calc_L2_schaden(delta1, delta2, t, t_max, K_wald, L1_holz, L2_holz, q1, L2_schadenintensity, tau) if tau > 0 and K_wald > 0 and L2_schaden == 0 else 0
    t_wald_exploited = f">{t_max}" if t_wald_exploited > t_max else f"={t_wald_exploited}"
    nash_product = U1_diskont * U2_diskont
    if constraints_failure < 0 and constraints_failure > -1e-6:
        constraints_failure = 0
    return constraints_failure, nash_product, K_wald, t_wald_exploited, values

# Cache um eine doppelte Berechnung des Nashproduktes durch scipy.optimize.minimize zu vermeiden (caching der Ergebnisse, 
# um calc_nash_product für die Optimierungsfunktion und die constraintsfunktion nicht doppelt zu berechnen).
class ScipyOptimizerCache:
    def __init__(self, delta1, delta2, t_max, L2_schadenintensity):
        self.delta1 = delta1
        self.delta2 = delta2
        self.t_max = t_max
        self.L2_schadenintensity = L2_schadenintensity
        self.cached_x = None
        self.cached_result = None
    # Führt calc_nash_product aus, falls vorher nicht mit x aufgerufen wurde
    def compute(self, x):
        if self.cached_x is None or not np.array_equal(x, self.cached_x):
            constraints_failure, nash_product, _, _, _ = calc_nash_product(x, self.delta1, self.delta2, self.t_max, self.L2_schadenintensity, export_values = False)
            self.cached_result = (constraints_failure, nash_product)
            self.cached_x = np.copy(x)
    # Rückgabe -1 * nash_product, entweder berechnet oder aus cache
    def objective(self, x):
        self.compute(x)
        return -self.cached_result[1]
    # Rückgabe constraints_failure, entweder berechnet oder aus cache
    def constraint(self, x):
        self.compute(x)
        return self.cached_result[0] 

# Nashprodukt mit Rückgabewert 0 falls constraints verletzt sind
def nash_product_grid(x, settings):
    constraints_failure, nash_product, K_wald, t_wald_exploited, values = calc_nash_product(x, settings.delta1, settings.delta2, settings.t_max, settings.L2_schadenintensity, export_values = False)
    return nash_product if constraints_failure >= 0 else 0

# Race to preempt am Beispiel eines gemeinschaftlichen Waldes
def rcw_02_race_to_preempt(settings, do_pretest_L2_holz = False, do_grid_search = True, do_numerical = True, do_plot = False):
    print(f"Robinson-Crusoe-Wirtschaft mit begrenztem Wald und Schadenspotential, δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, t_max={settings.t_max}, max.Schadenintensität={settings.L2_schadenintensity:.1f}:")
    sol_grid_x = None
    if do_pretest_L2_holz: # Kurztest zur Stabilitätskontrolle ausgewählter Lösungen mit L1_holz = 2.0 und L2_holz = 0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, ... usw. Test passed.
        best_test = None
        x_tests = [[2.0, 0.0], [2.0, 0.001], [2.0, 0.005], [2.0, 0.01], [2.0, 0.05], [2.0, 0.1], [2.0, 0.15], [2.0, 0.2], [2.0, 0.25], [2.0, 0.3], [2.0, 0.4], [2.0, 0.5], [2.0, 0.6], [2.0, 0.7], [2.0, 0.8], [2.0, 0.9], [2.0, 1.0], [2.0, 1.5], [2.0, 2.0]]
        for x_test in x_tests:
            constraints_failure, nash_product, K_wald, t_wald_exploited, values = calc_nash_product(x_test, settings.delta1, settings.delta2, settings.t_max, settings.L2_schadenintensity, export_values = True)
            if constraints_failure >= 0:
                print(f"  Kurztest: δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, x={fmt(x_test, '{:.3f}')}, Nashprod={nash_product:.2f}, K_wald={K_wald:.2f}, t_exploited{t_wald_exploited}, L2_schaden>0: {[f't={t}' for t in range(settings.t_max) if values.L2_schaden[t]>0]}")
                if best_test is None or best_test[3] < nash_product:
                    best_test = (x_test, settings.delta1, settings.delta2, nash_product, K_wald, t_wald_exploited, values)
        print(f"  Kurztest: δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, best_x={fmt(best_test[0], '{:.3f}')}, Nashprod={best_test[3]:.2f}, K_wald={best_test[4]:.2f}, t_exploited{best_test[5]}, L2_schaden>0: {[f't={t}' for t in range(settings.t_max) if best_test[6].L2_schaden[t]>0]}")
        if do_plot:
            rcw_02_plot(f"Kurztest Robinson-Crusoe-Wirtschaft mit begrenztem Wald und Schadenspotential\nδ1={best_test[1]:.2f}, δ2={best_test[2]:.2f}, L1_holz={best_test[0][0]:.3f}, L2_holz={best_test[0][1]:.3f}, NP={best_test[3]:.2f}", best_test[6])
    if do_grid_search: # Grid-search mit 2 Unbekannten
        start_time = time.perf_counter()
        sol_grid_x, sol_grid_np = rcw_grid_search_iter(func = nash_product_grid, func_arg = settings, grid_x_start = [0, 0], grid_x_stop = np.array([L_max, L_max]) + 1e-6, grid_x_steps = [1, 1], num_iter = 1, verbose=1, constraints_fun = None)
        constraints_failure, nash_product, K_wald, t_wald_exploited, values = calc_nash_product(sol_grid_x, settings.delta1, settings.delta2, settings.t_max, settings.L2_schadenintensity, export_values = True)
        execution_time_sec = time.perf_counter() - start_time
        if constraints_failure < 0:
            print(f"  ## grid-search(δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}) failed: constraints_failure={constraints_failure}, x={fmt(sol_grid_x, '{:.3f}')}, Nashprod={sol_grid_np:.2f}, K_wald={K_wald:.2f}, t_exploited{t_wald_exploited}, execution_time={execution_time_sec:.2f} sec")
        else:
            print(f"  grid-search(δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}): x={fmt(sol_grid_x, '{:.3f}')}, Nashprod={sol_grid_np:.2f}, K_wald={K_wald:.2f}, t_exploited{t_wald_exploited}, execution_time={execution_time_sec:.2f} sec")
            print(f"  L2_schaden>0: {[f't={t}' for t in range(settings.t_max) if values.L2_schaden[t]>0]}")
            if do_plot:
                rcw_02_plot(f"Robinson-Crusoe-Wirtschaft mit begrenztem Wald und Schadenspotential\nδ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, L1_holz={sol_grid_x[0]:.2f}, L2_holz={sol_grid_x[1]:.2f}", values)
    if do_numerical: # Numerische Lösung mit 2 Unbekannten und Startwerten aus grid-search
        # opt_x0s = [[2.0, 0.0], [2.0, 0.001], [2.0, 0.005], [2.0, 0.01], [2.0, 0.05], [2.0, 0.1], [2.0, 0.2]] # Stabilitätscheck mit Startwerten L1_holz = 2.0 und L2_holz = 0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, check done
        opt_x0s = [[2.0, 0.001]] # [[2.0, 0.0], sol_grid_x]
        opt_bounds = ((0, L_max), (0, L_max))
        for opt_x0 in opt_x0s:
            if opt_x0 is None: # i.e. sol_grid_x falls do_grid_search==False
                continue
            opt_cache = ScipyOptimizerCache(settings.delta1, settings.delta2, settings.t_max, settings.L2_schadenintensity)
            # opt_function = lambda x: -calc_nash_product(x, settings.delta1, settings.delta2, settings.t_max, settings.L2_schadenintensity, export_values = False)[1]
            opt_constraints = [scipy.optimize.NonlinearConstraint(fun = opt_cache.constraint, lb = 0, ub = np.inf)] # min(U1(t) - U1_autarkie(t), U2(t) - U2_autarkie(t)) >= 0
            for algo_cnt in range(1):
                try:
                    if algo_cnt == 0:
                        algo_str = "scipy.optimize.minimize"
                        sol_equilibrium = scipy.optimize.minimize(fun = opt_cache.objective, x0 = opt_x0, bounds = opt_bounds, constraints = opt_constraints, method = "SLSQP", options = {"maxiter": 1000, "ftol": 1e-6})
                    elif algo_cnt == 1:
                        algo_str = "scipy.optimize.shgo"
                        sol_equilibrium = scipy.optimize.shgo(func = opt_cache.objective, bounds = opt_bounds, sampling_method='sobol', constraints = opt_constraints) # options = {"maxiter": 1000, "f_tol": 1e-6, "f_min": 0}
                    else:
                        break
                    if sol_equilibrium.success:
                        sol_x, sol_np = sol_equilibrium.x, -sol_equilibrium.fun
                        constraints_failure, nash_product, K_wald, t_wald_exploited, values = calc_nash_product(sol_x, settings.delta1, settings.delta2, settings.t_max, settings.L2_schadenintensity, export_values = True)
                        if constraints_failure < 0:
                            print(f"  ## {algo_str} failed, constraints_failure={constraints_failure}, δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, x0={fmt(opt_x0, '{:.3f}')}, x={fmt(sol_x, '{:.3f}')}, Nashprod={nash_product:.2f}, K_wald={K_wald:.2f}, t_exploited{t_wald_exploited}")
                        else:
                            print(f"  {algo_str}(δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, x0={fmt(opt_x0, '{:.3f}')}): x={fmt(sol_x, '{:.3f}')}, Nashprod={sol_np:.2f}, K_wald={K_wald:.2f}, t_exploited{t_wald_exploited}")
                            print(f"  L2_schaden>0: {[f't={t}' for t in range(settings.t_max) if values.L2_schaden[t]>0]}")
                            if do_plot:
                                rcw_02_plot(f"Robinson-Crusoe-Wirtschaft mit begrenztem Wald und Schadenspotential\nδ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, L1_holz={sol_x[0]:.2f}, L2_holz={sol_x[1]:.2f}", values)
                    else:
                        print(f"  ## {algo_str} failed: δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, x0={fmt(opt_x0, '{:.3f}')}, \"{sol_equilibrium.message}\"")
                except Exception as exc:
                    exc_str = str(exc).split('\n')[0]
                    print(f"  ## Exception during {algo_str}: {exc_str}")
    if not do_grid_search and not do_numerical:  # Nur plot der numerischen Lösung
        sol_x = [2.0, 0.0]
        constraints_failure, nash_product, K_wald, t_wald_exploited, values = calc_nash_product(sol_x, settings.delta1, settings.delta2, settings.t_max, settings.L2_schadenintensity, export_values = True)
        if constraints_failure < 0:
            print(f"  ## constraints_failure={constraints_failure}, δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, x={fmt(sol_x, '{:.3f}')}, Nashprod={nash_product:.2f}, K_wald={K_wald:.2f}, t_exploited{t_wald_exploited}")
        else:
            print(f"  Numerische Lösung, δ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}: x={fmt(sol_x, '{:.3f}')}, Nashprod={nash_product:.2f}, K_wald={K_wald:.2f}, t_exploited{t_wald_exploited}")
            print(f"  L2_schaden>0: {[f't={t}' for t in range(settings.t_max) if values.L2_schaden[t]>0]}")
            if do_plot:
                rcw_02_plot(f"Robinson-Crusoe-Wirtschaft mit begrenztem Wald und Schadenspotential\nδ1={settings.delta1:.2f}, δ2={settings.delta2:.2f}, L1_holz={sol_x[0]:.2f}, L2_holz={sol_x[1]:.2f}", values)
    print(f"")
    return constraints_failure, nash_product, K_wald, t_wald_exploited, values

if __name__ == "__main__":

    # Debugging
    # constraints_failure, nash_product, K_wald, t_wald_exploited, values = calc_nash_product([2.000,0.200], 1, 1, 100, L_max, export_values = True)
    # rcw_02_plot_dbg(plots=[[values.t, np.array(values.L2_schaden)/L_max, "L2_schaden/L_max"], [values.t, np.array(values.K_wald)/K_wald_max, "K_wald/K_wald_max"],
    #                         [values.t, np.array(values.U1_t)/10, "U1/10"], [values.t, np.array(values.U2_t)/10, "U2/10"], [values.t, np.array(values.U1_autarkie)/10, "U1_autarkie/10"], [values.t, np.array(values.U2_autarkie)/10, "U2_autarkie/10"],
    #                         [values.t, values.tau2_alternativ, "tau2_alternativ"], [values.t, values.q1, "q1"]], xlabel="t", ylabel="", block=True, figsize=(15,5))
    # Debugging
    # delta1, delta2, t_max, L2_schadenintensity = 1, 1, 100, L_max
    # x = np.array([2.0, 0.05, 1.5]) # x[0] = L1_holz = 2.00, x[1] = L2_holz = 0.05, x[2] = tau_thresh = 1.50
    # print(f"Race to preempt mit begrenztem Wald, debug mit δ1={delta1:.2f}, δ2={delta2:.2f}, L1_holz={x[0]:.2f}, xL2_holz={x[1]:.2f}, tau_thresh={x[2]:.2f}:")
    # t_values = np.arange(0, t_max)
    # U1_autarkie_t, U2_autarkie_t, K_wald_t, q1_t, tau2_alt_t, tau_t, L2_schaden_t = [], [], [], [], [], [], []
    # K_wald = K_wald_max
    # L2_schaden = 0
    # q1 = 1
    # for t in t_values:
    #     # Waldentwicklung
    #     K_wald, L1_holz, L2_holz, L2_schaden, Y_holz = calc_K_wald_update(K_wald, L1_holz=x[0], L2_holz=x[1], L2_schaden=L2_schaden)
    #     # Robinsons Einschätzung 0 <= q1(t) <= 1, wie glaubwürdig Freitags Drohung ist (erwartete Schadensintensität):
    #     q1 = calc_q1(t, q1, L2_schaden, L2_schadenintensity)
    #     # Autarkie-Nutzen unter Annahme maximaler Abholzung durch den jeweils anderen (Robinson nimmt maximale Abholzung durch Freitag mit L2_schaden=L_max an, Freitag nimmt maximale Abholzung durch Robinson mit L1_holz=L_max an)
    #     U1_autarkie_val, U2_autarkie_val = calc_U_autarkie(t, t_max, K_wald, Y_holz, q1, delta1, delta2, L_max, L_max, L_max)
    #     # Tauschanteil tau2_alternativ und Tauschpreis tau
    #     tau2_alternativ, tau = calc_tau2_alternativ(L1_holz, L2_holz, L2_schaden, U1_autarkie_val, U2_autarkie_val)
    #     # Plot über Zeit t
    #     U1_autarkie_t.append(U1_autarkie_val)
    #     U2_autarkie_t.append(U2_autarkie_val)
    #     K_wald_t.append(K_wald/K_wald_max)
    #     q1_t.append(q1)
    #     tau2_alt_t.append(tau2_alternativ)
    #     tau_t.append(tau)
    #     L2_schaden_t.append(L2_schaden)
    #     # Ist der Preis zu hoch, reagiert Freitag auf Robinsons Tauschverhalten mit Abholzung zum Schaden aller. 
    #     L2_schaden = calc_L2_schaden(delta1, delta2, t, t_max, K_wald, L1_holz, L2_holz, q1, L2_schadenintensity, tau) if tau > 0 and K_wald > 0 and L2_schaden == 0 else 0
    # rcw_02_plot_dbg([(t_values, K_wald_t), (t_values, q1_t), (t_values, L2_schaden_t), (t_values, U1_autarkie_t), (t_values, U2_autarkie_t), (t_values, tau2_alt_t), (t_values, tau_t)], "Zeit t", "Kapital K, Glaubwürdigkeit q, Schaden, Nutzen U, Tauschanteil/-verhältnis tau", block=False)

    # Race to preempt am Beispiel eines gemeinschaftlichen Waldes:
    do_pretest_L2_holz = False # Kurztest zur Stabilitätskontrolle mit L1_holz = 2.0 und L2_holz = 0.0, 0.01, 0.05, 0.1, 0.15, ... usw.
    do_grid_search = True      # grid search, sehr rechenintensiv, Plausibilitätscheck und Startwerte der numerischen Lösung
    do_numerical = True        # Numerische Lösung mit scipy.optimize.minimize, ebenfalls sehr rechenintensiv
    do_plot = True             # Falls do_grid_search = False und do_numerical = False, dann nur die numerische Lösung plotten (schnell und ohne aufwendige Berechnungen)
    rcw_02_race_to_preempt(RCWSettings(delta1 = 1, delta2 = 1, t_max = 100, L2_schadenintensity = L_max), do_pretest_L2_holz, do_grid_search, do_numerical, do_plot)
    rcw_02_race_to_preempt(RCWSettings(delta1 = 1, delta2 = 1, t_max = 115, L2_schadenintensity =   0.1), do_pretest_L2_holz, do_grid_search, do_numerical, do_plot)
    print(f"mini_wm_robinson_race_to_preempt finished.")
    plt.show()
