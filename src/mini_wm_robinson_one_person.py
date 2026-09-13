"""
Mini-WM: Ein minimalistisches Wirtschaftsmodell,
Teil 1: Robinson-Crusoe-Wirtschaft mit 1 Person.
Erläuterungen siehe MiniWM_Teil01_RobinsonCrusoe.md.
"""
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

# Speichert den aktuellen plot als png file
def save_plot_png(pngfile):
    if pngfile != "":
        pathlib.Path(pngfile).unlink(missing_ok=True)
        plt.savefig(pngfile)
        print(f"  Plot gespeichert in {pngfile}")

# Robinson-Crusoe-Wirtschaft, 1-Personen-Subsistenzwirtschaft: Plot der Transformationskurve (production-possibility curve, PPC) und Indifferenzkurve für Fische und Kokosnüsse
def rcw_01_plot_ppc(pngfiles = ["",""], figsize=(5, 5)):
    print(f"Robinson-Crusoe-Wirtschaft, 1-Personen-Subsistenzwirtschaft: Transformationskurve und Indifferenzkurve\n")
    # Transformationskurve:
    # Produktion Y_nuss(L) = A_nuss * (L ** alpha_nuss) mit A_nuss = 30 / 12**0.5, alpha_nuss = 0.5 und Arbeitseinsatz L
    # Produktion Y_fisch(L) = A_fisch * (L ** alpha_fisch) mit A_fisch = 20 / 12**0.5, alpha_fisch = 0.5 und Arbeitseinsatz L
    fig = plt.figure(figsize=figsize)
    A_nuss = (30 / (12**0.5))
    A_fisch = (20 / (12**0.5))
    alpha_nuss = 0.5
    alpha_fisch = 0.5
    Y_nuss = lambda L : A_nuss * (L ** alpha_nuss)
    Y_fisch = lambda L : A_fisch * (L ** alpha_fisch)
    MSR = 2 # Verhältnis der Präferenz für Fisch gegenüber Nüssen: 1 Fisch entspricht 2 Nüssen im Nutzen
    for L_max in [4, 6, 8, 10, 12]:
        L_min =  0.0
        Y_nuss_values = []
        Y_fisch_values = []
        for i, L_ratio in enumerate(np.linspace(0.0, 1.0, 1000)):
            L_nuss = L_min + L_ratio * (L_max - L_min)
            L_fisch = L_min + (1-L_ratio) * (L_max - L_min)
            assert(math.isclose(L_nuss + L_fisch, L_max, rel_tol=1.0e-6, abs_tol=1.0e-6))
            Y_nuss_values.append(Y_nuss(L_nuss))
            Y_fisch_values.append(Y_fisch(L_fisch))
        plt.plot(Y_fisch_values, Y_nuss_values, label=f"L={L_max}")
    plt.xlabel("Produktionsmenge Fische")
    plt.ylabel("Produktionsmenge Nüsse")
    plt.legend()
    plt.title("Robinson-Crusoe-Wirtschaft: Transformationskurve")
    save_plot_png(pngfiles[0])
    # Indifferenzkurve:
    # Nutzen U_nuss(L) = C_nuss(L)
    # Nutzen U_fisch(L) = MSR * C_fisch(L)
    # Gesamtnutzen: U = U_nuss + U_fisch = C_nuss + MSR * C_fisch <=> C_nuss = U - MSR * C_fisch
    fig = plt.figure(figsize=figsize)
    for U in [40, 50, 60, 70]:
        C_fisch = np.linspace(0, U/2, 1000)
        C_nuss = U - MSR * C_fisch
        plt.plot(C_fisch, C_nuss, label=f"U={U}")
    plt.xlabel("Konsummenge Fische")
    plt.ylabel("Konsummenge Nüsse")
    plt.legend()
    plt.title("Robinson-Crusoe-Wirtschaft: Indifferenzkurve")
    save_plot_png(pngfiles[1])
    # Maximierung der Nutzenfunktion U = U_fisch + U_nuss = MSR * C_fisch + C_nuss = MSR * Y_fisch(L_fisch) + Y_nuss(L_nuss) = 2 * 5.77 * L_fisch**(0.5) + 8.66 * L_nuss**(0.5) mit L_fisch + L_nuss = L:
    # Gesucht ist die Lösung für ∂U(x,y)/∂x = 0 mit der Nebenbedingung F(x,y) = 0
    # Implizites Differenzieren liefert dy/dx mit F(x,y) = 0 ⇔ (∂F/∂x)dx + (∂F/∂y)dy = 0 ⇔ dy/dx = -(∂F/∂x) / (∂F/∂y) 
    # Das totale Differential liefert ∂U(x,y)/∂x = ∂U/∂x + ∂U/∂y * dy/dx
    # Im Fall der Robinson-Crusoe-Wirtschaft mit x = L_fisch, y = L_nuss also:
    x, y, L = symbols("x y L", real = True)
    U = MSR * A_fisch * (x ** alpha_fisch) + A_nuss * (y ** alpha_nuss)
    F = x + y - L
    dydx = -diff(F, x) / diff(F, y)
    dUdx = diff(U, x)
    dUdy = diff(U, y)
    dUdx_total = dUdx + dUdy * dydx
    sol_x = solve(Eq(dUdx_total, 0), x)
    sol_y = solve(Eq(dUdx_total, 0), y)
    sol_x = sol_x[0] if len(sol_x) == 1 else sol_x
    sol_y = sol_y[0] if len(sol_y) == 1 else sol_y
    print(f"  Maximierung der Nutzenfunktion U = {U} mit Nebenbedingung F = {F} = 0:")
    print(f"    dy/dx = {dydx}")
    print(f"    ∂U/∂x = {dUdx}")
    print(f"    ∂U/∂y = {dUdy}")
    print(f"    ∂U(x,y)/∂x = {dUdx_total}")
    print(f"    ∂U(x,y)/∂x = 0 <=> x = {sol_x} <=> y = {sol_y}")
    print(f"")

# Robinson-Crusoe-Wirtschaft, 1-Personen-Subsistenzwirtschaft: Gleichgewicht zwischen Produktion und Konsum
def rcw_01_equilibrium(pngfiles = [""]):
    # Modell:
    # * Produzierte Menge $Y_{fisch} = A_{fisch} \cdot L_{fisch}^{α_{fisch}}$ und $Y_{nuss} = A_{nuss} \cdot L_{nuss}^{α_{nuss}}$
    # * Konsumierte Menge $C_{fisch} = C_{fisch,subsistenz} + C_{fisch,genuss}$ und $C_{nuss} = C_{nuss,subsistenz} + C_{nuss,genuss}$
    # * Nutzen $U = U_{subsistenz}(C_{fisch,subsistenz}, C_{nuss,subsistenz}, L_{subsistenz}) + U_{genuss}(C_{fisch,genuss}, C_{nuss,genuss}, L_{genuss})$
    # Im Gleichgewicht gilt: 
    # * Ressourcenrestriktion: $C_{fisch,subsistenz} + C_{fisch,genuss} + I_{fisch} = Y_{fisch}$ und $C_{nuss,subsistenz} + C_{nuss,genuss} + I_{nuss} = Y_{nuss}$
    # * Zeitrestriktion: $F = L_{fisch} + L_{nuss} + L_{subsistenz} + L_{genuss} + L_{invest} - 24 = 0$
    # * Subsistenzrestriktion: $L_{subsistenz} ≥ L_{minimum}$, z.B. $L_{subsistenz} ≥ 10$
    # Shortcuts: x1 = L_fisch, x2 = L_nuss, x3 = L_subsistenz, x4 = L_genuss, x5 = L_invest
    print(f"Robinson-Crusoe-Wirtschaft, 1-Personen-Subsistenzwirtschaft: Gleichgewicht zwischen Produktion und Konsum\n")

    # Produzierte Menge $Y_{fisch} = A_{fisch} \cdot L_{fisch}^{α_{fisch}}$ und $Y_{nuss} = A_{nuss} \cdot L_{nuss}^{α_{nuss}}$
    x1, x2 = symbols("L_fisch L_nuss", real=True, positive=True)
    A_fisch = 20 / (12**0.5)
    A_nuss = 30 / (12**0.5)
    alpha_fisch = 0.5
    alpha_nuss = 0.5
    Y_fisch = A_fisch * (x1**alpha_fisch)
    Y_nuss = A_nuss * (x2**alpha_nuss)

    # Zeitrestriktion: $F = L_{fisch} + L_{nuss} + L_{subsistenz} + L_{genuss} + L_{invest} - 24 = 0$
    L_sum = 24

    # Subsistenz-Nutzen: $U_{subsistenz} = A_{subsistenz} \cdot (MSR \cdot C_{fisch,subsistenz} + C_{nuss,subsistenz})^{α_{subsistenz}} \cdot L_{subsistenz}^{α_{subsistenz}}$
    MSR = 2 # Verhältnis der Präferenz für Fisch gegenüber Nüssen: 1 Fisch entspricht 2 Nüssen im Nutzen
    A_subsistenz = 1 / ((MSR * 10 + 20)**2 * 10**2)
    alpha_subsistenz = 2 # Subsistenz mit höherer Elastizität

    # Genuss-Nutzen :$U_{genuss} = A_{genuss} \cdot (MSR \cdot C_{fisch,genuss} + C_{nuss,genuss})^{α_{genuss}} \cdot L_{genuss}^{α_{genuss}}$
    # U_subsistenz(10,20,11) = U_subsistenz mit 10 Fischen, 20 Kokosnüssen und 11 Stunden Freizeit
    # U_subsistenz(10,20,11) = U_genuss(3,4,5) = U_genuss mit 3 Fischen, 4 Kokosnüssen und 5 Stunden Freizeit
    # => U_genuss(3,4,5) = A_genuss * ((2 * 3 + 4)**0.2 * 5**0.2) = U_subsistenz(10,20,11) => A_genuss = U_subsistenz(10,20,11) / ((2 * 3 + 4)**0.2 * 5**0.2)
    alpha_genuss = 0.2
    U_subsistenz_10_20_11 = A_subsistenz * (MSR * 10 + 20)**2 * 11**2 # U_subsistenz mit 10 Fischen, 20 Kokosnüssen und 11 Stunden Freizeit
    A_genuss = U_subsistenz_10_20_11 / ((MSR * 3 + 4)**alpha_genuss * 5**alpha_genuss) # A_genuss aus U_subsistenz(10,20,11) = U_genuss(3,4,5)

    # Fall 1, Gleichgewicht bei reiner Subsistenz, U = U_subsistenz(x1,x2,x3): Plausibilitätscheck und Startwerte für nsolve durch Einsetzen von Werten grid-search
    L_step = 0.1
    best_grid = [0]
    for x1_grid in np.arange(0, L_sum, L_step):     # x1 = L_fisch
        for x2_grid in np.arange(0, L_sum, L_step): # x2 = L_nuss
            x3_grid = L_sum - x1_grid - x2_grid     # x3 = L_subsistenz = 24 - x1 - x2
            if x3_grid > 0:
                Y_fisch_grid = A_fisch * (x1_grid**alpha_fisch) # identisch aber schneller als Y_fisch.subs(x1, x1_grid)
                Y_nuss_grid = A_nuss * (x2_grid**alpha_nuss)    # identisch aber schneller als Y_nuss.subs(x2, x3_grid)
                U_subsistenz_grid = A_subsistenz * ((MSR*Y_fisch_grid+Y_nuss_grid)**alpha_subsistenz) * (x3_grid**alpha_subsistenz) # Gleichgewicht: Y_fisch=C_{fisch,subsistenz}, Y_nuss=C_{nuss,subsistenz}
                if best_grid[-1] < U_subsistenz_grid:
                    best_grid = [x1_grid, x2_grid, x3_grid, U_subsistenz_grid]
    print(f"  Gleichgewicht bei reiner Subsistenz, Plausibilitätscheck (grid-search): x_grid = [{best_grid[0]:.1f}, {best_grid[1]:.1f}, {best_grid[2]:.1f}], U_subsistenz = {best_grid[-1]:.3f}")
    
    # Fall 1, Gleichgewicht bei reiner Subsistenz, U = U_subsistenz(x1,x2,x3): Lösung dU/dx = 0 mittels Lagrange
    x3 = symbols("L_subsistenz", real=True, positive=True)
    U_subsistenz = A_subsistenz * ((MSR*Y_fisch+Y_nuss)**alpha_subsistenz) * (x3**alpha_subsistenz) # Gleichgewicht: Y_fisch=C_{fisch,subsistenz}, Y_nuss=C_{nuss,subsistenz}
    # Lagrange: L = U(x1, x2, x3) + λ*(24 - x1 - x2 - x3)
    # First-Order-Conditions (FOCs): ∂L/∂x1 = 0, ∂L/∂x2 = 0, ∂L/∂x3 = 0, ∂L/∂λ = 0
    L_lambda = symbols("lambda", real=True, positive=True)
    L = U_subsistenz + L_lambda * (L_sum - x1 - x2 - x3) # Zeitrestriktion: $F = L_{fisch} + L_{nuss} + L_{subsistenz} - 24 = 0$
    dL_dx1 = simplify(diff(L, x1))
    dL_dx2 = simplify(diff(L, x2))
    dL_dx3 = simplify(diff(L, x3))
    dL_dlambda = simplify(diff(L, L_lambda))
    sol_equilibriums = nsolve([dL_dx1, dL_dx2, dL_dx3, dL_dlambda], [x1, x2, x3, L_lambda], [best_grid[0], best_grid[1], best_grid[2], 1])
    assert(sol_equilibriums.shape == (4,1)) # Eindeutige Lösung erwartet
    sol_x1, sol_x2, sol_x3 = sol_equilibriums[0:3]
    sol_subs = {x1: sol_x1, x2: sol_x2, x3: sol_x3}
    sol_Y_fisch = Y_fisch.subs(sol_subs)
    sol_Y_nuss = Y_nuss.subs(sol_subs)
    sol_U = U_subsistenz.subs(sol_subs)
    print(f"  Gleichgewicht bei reiner Subsistenz, Lagrange-Lösung = [{sol_x1:.3f}, {sol_x2:.3f}, {sol_x3:.3f}], U_subsistenz = {sol_U:.3f}")
    print(f"  Gleichgewicht bei reiner Subsistenz: {x1} = {sol_x1:.3f}, {x2} = {sol_x2:.3f}, {x3} = {sol_x3:.3f}, {x1}/{x2} = {sol_x1/sol_x2:.3f}, Y_fisch = {sol_Y_fisch:.3f}, Y_nuss = {sol_Y_nuss:.3f}, U_subsistenz = {sol_U:.3f}")
    print(f"")

    # Fall 2, Gleichgewicht bei Subsistenz und Genusskonsum, U = U_subsistenz(x1,x2,x3) + U_genuss(x1,x2,x4): Plausibilitätscheck und Startwerte für nsolve durch Einsetzen von Werten grid-search
    L_step = 2.0
    Y_step = 2.0
    best_grid = [0]
    x1_grid_range = np.arange(0, L_sum, L_step)
    x2_grid_range = np.arange(0, L_sum, L_step)
    x3_grid_range = np.arange(0, L_sum, L_step)
    for step_iter in range(4):
        for x1_grid in x1_grid_range:          # x1 = L_fisch
            for x2_grid in x2_grid_range:      # x2 = L_nuss
                for x3_grid in x3_grid_range:  # x3 = L_subsistenz
                    x4_grid = L_sum - x1_grid - x2_grid - x3_grid # x4 = L_genuss = 24 - x1 - x2 - x3
                    if x4_grid <= 0:
                        continue
                    Y_fisch_grid = A_fisch * (x1_grid**alpha_fisch) # identisch aber schneller als Y_fisch.subs(x1, x1_grid)
                    Y_nuss_grid = A_nuss * (x2_grid**alpha_nuss)    # identisch aber schneller als Y_nuss.subs(x2, x3_grid)
                    # Gleichgewicht: Y_fisch = C_{fisch,subsistenz} + C_{fisch,genuss}, Y_nuss = C_{nuss,subsistenz} + C_{nuss,genuss}
                    for C_fisch_subsistenz in np.arange(0, Y_fisch_grid + Y_step, Y_step):
                        C_fisch_genuss = Y_fisch_grid - C_fisch_subsistenz
                        if C_fisch_genuss <= 0:
                            continue
                        for C_nuss_subsistenz in np.arange(0, Y_nuss_grid + Y_step, Y_step):
                            C_nuss_genuss = Y_nuss_grid - C_nuss_subsistenz
                            if C_nuss_genuss <= 0:
                                continue
                            U_subsistenz_grid = A_subsistenz * ((MSR*C_fisch_subsistenz+C_nuss_subsistenz)**alpha_subsistenz) * (x3_grid**alpha_subsistenz) 
                            U_genuss_grid = A_genuss * ((MSR*C_fisch_genuss+C_nuss_genuss)**alpha_genuss) * (x4_grid**alpha_genuss)
                            U_grid = U_subsistenz_grid + U_genuss_grid
                            if best_grid[-1] < U_grid:
                                best_grid = [x1_grid, x2_grid, x3_grid, x4_grid, C_fisch_subsistenz, C_nuss_subsistenz, U_subsistenz_grid, C_fisch_genuss, C_nuss_genuss, U_genuss_grid, U_grid ]
        x1_grid_range = np.arange(max(best_grid[0] - L_step, 0), min(best_grid[0] + L_step, L_sum), 0.5 * L_step)
        x2_grid_range = np.arange(max(best_grid[1] - L_step, 0), min(best_grid[1] + L_step, L_sum), 0.5 * L_step)
        x3_grid_range = np.arange(max(best_grid[2] - L_step, 0), min(best_grid[2] + L_step, L_sum), 0.5 * L_step)
        L_step = 0.5 * L_step
        Y_step = 0.5 * Y_step
    print(f"  Gleichgewicht bei Subsistenz und Genusskonsum, Plausibilitätscheck (grid-search): x_grid = [{best_grid[0]:.1f}, {best_grid[1]:.1f}, {best_grid[2]:.1f}, {best_grid[3]:.1f}], U_subsistenz = {best_grid[6]:.3f}, U_genuss = {best_grid[9]:.3f}, U = {best_grid[10]:.3f}")

    # Fall 2, Gleichgewicht bei Subsistenz und Genusskonsum, U = U_subsistenz(x1,x2,x3) + U_genuss(x1,x2,x4): Lösung dU/dx = 0 mittels Lagrange
    # U_subsistenz = A_subsistenz * (C_subsistenz**alpha_subsistenz) * (x3**alpha_subsistenz), C_subsistenz = (MSR*C_fisch_subsistenz+C_nuss_subsistenz)
    # U_genuss = A_genuss * (C_genuss**alpha_genuss) * (x4**alpha_genuss), C_genuss = (MSR*C_fisch_genuss+C_nuss_genuss)
    # Gleichgewicht: C_subsistenz + C_genuss = MSR*C_fisch_subsistenz + C_nuss_subsistenz + MSR*C_fisch_genuss + C_nuss_genuss = 2 * Y_fisch + Y_nuss
    # => Das Gleichungsssystem hat (ausser x1, x2, x3, x4) nur 1 Unbekannte statt 2 zusätzliche Unbekannten:
    # C_subsistenz = symbols("C_subsistenz", real=True, positive=True), C_genuss = MSR * Y_fisch + Y_nuss - C_subsistenz
    x4, C_subsistenz = symbols("L_genuss C_subsistenz", real=True, positive=True) # x4 = L_genuss, C_subsistenz = MSR*C_fisch_subsistenz+C_nuss_subsistenz
    C_genuss = MSR * Y_fisch + Y_nuss - C_subsistenz # Gleichgewicht: C_genuss = MSR * Y_fisch + Y_nuss - C_subsistenz
    U_subsistenz = A_subsistenz * (C_subsistenz**alpha_subsistenz) * (x3**alpha_subsistenz) 
    U_genuss = A_genuss * (C_genuss**alpha_genuss) * (x4**alpha_genuss)
    # Lagrange: L = U(x1, x2, x3, x4, Cs) + λ*(24 - x1 - x2 - x3 - x4) mit x1 = L_fisch, x2 = L_nuss, x3 = L_subsistenz, x4 = L_genuss
    # First-Order-Conditions (FOCs): ∂L/∂x1 = 0, ∂L/∂x2 = 0, ∂L/∂x3 = 0, ∂L/∂x4 = 0, ∂L/∂Cs = 0, ∂L/∂λ = 0
    L_lambda = symbols("lambda", real=True, positive=True)
    L = U_subsistenz + U_genuss + L_lambda * (L_sum - x1 - x2 - x3 - x4) # Zeitrestriktion: $F = L_fisch + L_nuss + L_subsistenz + L_genuss - 24 = 0$
    dL_dx1 = simplify(diff(L, x1))
    dL_dx2 = simplify(diff(L, x2))
    dL_dx3 = simplify(diff(L, x3))
    dL_dx4 = simplify(diff(L, x4))
    dL_dCs = simplify(diff(L, C_subsistenz))
    dL_dlambda = simplify(diff(L, L_lambda))
    sol_equilibriums = nsolve([dL_dx1, dL_dx2, dL_dx3, dL_dx4, dL_dCs, dL_dlambda], 
                              [x1, x2, x3, x4, C_subsistenz, L_lambda], 
                              [best_grid[0], best_grid[1], best_grid[2], best_grid[3], 2*best_grid[4]+best_grid[5], 1])
    sol_x1, sol_x2, sol_x3, sol_x4, sol_C_subsistenz = sol_equilibriums[0:5]
    sol_subs = {x1: sol_x1, x2: sol_x2, x3: sol_x3, x4: sol_x4, C_subsistenz: sol_C_subsistenz}
    sol_Y_fisch = Y_fisch.subs(sol_subs)
    sol_Y_nuss = Y_nuss.subs(sol_subs)
    sol_U_subsistenz = U_subsistenz.subs(sol_subs)
    sol_U_genuss = U_genuss.subs(sol_subs)
    sol_U = sol_U_subsistenz + sol_U_genuss
    print(f"  Gleichgewicht bei Subsistenz und Genusskonsum, Lagrange-Lösung = [{sol_x1:.3f}, {sol_x2:.3f}, {sol_x3:.3f}, {sol_x4:.3f}], U_subsistenz = {sol_U_subsistenz:.3f}, U_genuss = {sol_U_genuss:.3f}, U = {sol_U:.3f}")
    print(f"  Gleichgewicht bei Subsistenz und Genusskonsum: {x1} = {sol_x1:.3f}, {x2} = {sol_x2:.3f}, {x3} = {sol_x3:.3f}, {x4} = {sol_x4:.3f}, {x1}/{x2} = {sol_x1/sol_x2:.3f}, Y_fisch = {sol_Y_fisch:.3f}, Y_nuss = {sol_Y_nuss:.3f}, U_subsistenz = {sol_U_subsistenz:.3f}, U_genuss = {sol_U_genuss:.3f}, U = {sol_U:.3f}")
    print(f"")

    # Fall 3, Gleichgewicht bei Subsistenz, Genusskonsum und Investition in Kokosnussanbau:
    # Investition in Kokosnüsse: I_nuss = A_nuss_invest * (C_nuss_invest * L_nuss_invest)^alpha_nuss_invest
    # L_nuss_invest: Investitionszeit in Kokosnussanbau
    # C_nuss_invest = 10 * L_nuss_invest: Robinson kann in 1 Stunde Arbeit 10 Kokosnüsse einpflanzen
    alpha_nuss_invest = 1.0 # Elastizität 1: doppelte Investition ergibt doppelte Menge
    A_nuss_invest = 0.5 # 1 Kokosnuss investiert ergibt 0.5 Kokosnüsse mehr pro Stunde Arbeit in der Zukunft (A_nuss_invest < 1, da Investition nicht 100% effizient, Ernteverluste, Unwetter, etc.)
    # I_nuss = A_nuss_invest * ((C_nuss_invest * L_nuss_invest)**alpha_nuss_invest) # Investition in Kokosnüsse, z.B. 10 Kokosnüsse und 1 Stunde Arbeit ergeben 5 Kokosnüsse mehr pro Stunde Arbeit in der Zukunft
    # K_nuss = lambda t : K_nuss(t-1) + I_nuss if t > 0 else 0 # Kokosnuss-Kapital K_nuss(t+1) = K_nuss(t) + I_nuss
    # Nutzenfunktion U = U_subsistenz + U_genuss bleibt unverändert
    # Gleichgewicht (Ressourcenrestriktion): Y = C_subsistenz + C_genuss + I_nuss 
    # Gleichgewicht (Ressourcenrestriktion): Y_fisch = C_fisch_subsistenz + C_fisch_genuss, Y_nuss = C_nuss_subsistenz + C_nuss_genuss + C_nuss_invest
    # Gleichgewicht (Zeitrestriktion): F = L_fisch + L_nuss + L_subsistenz + L_genuss + L_nuss_invest - 24 = 0
    # Gleichgewicht: C_subsistenz + C_genuss = MSR C_fisch_subsistenz + C_nuss_subsistenz + MSR C_fisch_genuss + C_nuss_genuss = MSR Y_fisch + Y_nuss - C_nuss_invest}
    delta = 0.5 # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert.
    K_max = 100 # logistische Kapitalgrenze durch begrenzte Anbaufläche: K_nuss(t) = K_nuss(t-1) + I_nuss * (1 - K_nuss(t-1) / K_max)
    g_A = 0.01  # Produktivitätssteigerung durch Technologie: A(t) = A_0 (1 + g_A)^t, z.B. 1% Produktivitätssteigerung in jedem Zeitschritt

    # Plausibilitätscheck und Startwerte für nsolve durch Einsetzen von Werten grid-search
    print(f"  Gleichgewicht bei Subsistenz, Genusskonsum und Investition, Kapitalgrenze K_max={K_max}, Produktivitätssteigerung g_A={g_A*100}%, Plausibilitätscheck (grid-search):")
    best_grid = [0]
    L_step = 4.0
    x1_grid_range = np.arange(0, L_sum/2, L_step)
    x2_grid_range = np.arange(0, L_sum/2, L_step)
    x3_grid_range = np.arange(0, L_sum/2, L_step)
    for step_iter in range(3):
        for x1_grid in x1_grid_range:             # x1 = L_fisch
            for x2_grid in x2_grid_range:         # x2 = L_nuss
                for x3_grid in x3_grid_range:     # x3 = L_subsistenz
                    for ratio_fisch_subsistenz in np.linspace(0.0, 1.0, 10): 
                        # Robinson teilt seine Fischproduktion auf in Konsum für Subsistenz und Genuss:
                        # C_fisch_subsistenz = Y_fisch * ratio_fisch_subsistenz, C_fisch_genuss = Y_fisch * (1-ratio_fisch_subsistenz)
                        for ratio_nuss_subsistenz in np.linspace(0.0, 1.0, 10): 
                            # Robinson teilt seine Nussproduktion auf in Konsum für Subsistenz und Genuss+Investition:
                            # C_nuss_subsistenz = Y_nuss * ratio_nuss_subsistenz, C_nuss_genuss_invest = Y_nuss * (1-ratio_nuss_subsistenz)
                            for ratio_nuss_invest in np.linspace(0.0, 1.0, 10): 
                                # Robinson teilt seine Nussproduktion auf in Konsum für Subsistenz, Genuss und Investition:
                                # C_nuss_invest = C_nuss_genuss_invest * ratio_nuss_invest, C_nuss_genuss = C_nuss_genuss_invest * (1-ratio_nuss_invest)
                                K_nuss = 1 # Kokosnuss-Kapital K_nuss(t+1) = K_nuss(t) + I_nuss
                                U_sum = 0
                                for t in np.arange(0, 5, 1):
                                    # Gleichgewicht (Ressourcenrestriktion): Y_fisch = C_fisch_subsistenz + C_fisch_genuss
                                    Y_fisch_grid = A_fisch * ((1 + g_A)**t) * (x1_grid**alpha_fisch)
                                    C_fisch_subsistenz = Y_fisch_grid * ratio_fisch_subsistenz
                                    C_fisch_genuss = Y_fisch_grid * (1 - ratio_fisch_subsistenz)
                                    # Ohne Investition: Y_nuss = A_nuss * L_nuss^α_nuss, mit Investition: Y_nuss(t) = A_nuss (L_nuss * K_nuss(t))^α_nuss
                                    Y_nuss_grid = A_nuss * ((1 + g_A)**t) * ((x2_grid * K_nuss)**alpha_nuss)
                                    # Gleichgewicht (Ressourcenrestriktion): Y_nuss = C_nuss_subsistenz + C_nuss_genuss + C_nuss_invest
                                    C_nuss_subsistenz = Y_nuss_grid * ratio_nuss_subsistenz
                                    C_nuss_genuss = Y_nuss_grid * (1-ratio_nuss_subsistenz) * (1 - ratio_nuss_invest)
                                    C_nuss_invest = Y_nuss_grid * (1-ratio_nuss_subsistenz) * ratio_nuss_invest
                                    # Ressourcenrestriktion: Robinson kann in 1 Stunde Arbeit 10 Kokosnüsse einpflanzen
                                    x5_grid = C_nuss_invest / 10 # x5 = L_nuss_invest, C_nuss_invest/L_nuss_invest = 10
                                    # Zeitrestriktion: x4 = L_genuss = 24 - x1 - x2 - x3 - x5
                                    x4_grid = 24 - x1_grid - x2_grid - x3_grid - x5_grid
                                    if x4_grid < 0 or x5_grid < 0:
                                        U_sum = 0
                                        break
                                    # Investition I_nuss = A_nuss_invest * (C_nuss_invest**alpha_nuss_invest)
                                    I_nuss = A_nuss_invest * (C_nuss_invest**alpha_nuss_invest)
                                    # Nutzenfunktion U = U_subsistenz + U_genuss bleibt unverändert
                                    U_subsistenz_grid = A_subsistenz * ((MSR*C_fisch_subsistenz+C_nuss_subsistenz)**alpha_subsistenz) * (x3_grid**alpha_subsistenz) 
                                    U_genuss_grid = A_genuss * ((MSR*C_fisch_genuss+C_nuss_genuss)**alpha_genuss) * (x4_grid**alpha_genuss)
                                    # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
                                    U_sum += (delta ** t) * (U_subsistenz_grid + U_genuss_grid)
                                    # Ohne Kapitalbegrenzung (Robinson wohnt auf einer unendlich grossen Insel): K_nuss(t) = K_nuss(t-1) + I_nuss
                                    # K_nuss = K_nuss + I_nuss
                                    # Mit logistischer Kapitalgrenze durch begrenzte Anbaufläche: K_nuss(t) = K_nuss(t-1) + I_nuss * (1 - K_nuss(t-1) / K_max)
                                    K_nuss = K_nuss + I_nuss * (1 - K_nuss / K_max)
                                if best_grid[-1] < U_sum:
                                    best_grid = [x1_grid, x2_grid, x3_grid, x4_grid, x5_grid, Y_fisch_grid, Y_nuss_grid, ratio_fisch_subsistenz, ratio_nuss_subsistenz, ratio_nuss_invest, K_nuss, U_sum]
        x1_grid_range = np.arange(max(best_grid[0] - L_step, 0), min(best_grid[0] + L_step, L_sum), 0.5 * L_step)
        x2_grid_range = np.arange(max(best_grid[1] - L_step, 0), min(best_grid[1] + L_step, L_sum), 0.5 * L_step)
        x3_grid_range = np.arange(max(best_grid[2] - L_step, 0), min(best_grid[2] + L_step, L_sum), 0.5 * L_step)
        L_step = 0.5 * L_step
    print(f"    L_fisch={best_grid[0]:.1f}, L_nuss={best_grid[1]:.1f}, L_subsistenz={best_grid[2]:.1f}, L_genuss={best_grid[3]:.1f}, L_invest = {best_grid[4]:.1f}, Y_fisch = {best_grid[5]:.1f}, Y_nuss = {best_grid[6]:.1f}, K_nuss = {best_grid[-2]:.1f}, U_sum = {best_grid[-1]:.1f}")
    
    # Fall 3, Gleichgewicht bei Subsistenz, Genusskonsum und Investition in Kokosnussanbau: Lösung mit scipy.optimize.minimize und Startwerten aus grid-search
    # Gleichungssystem mit 6 Unbekannten: x1 = L_fisch, x2 = L_nuss, x3 = L_subsistenz, rfs = ratio_fisch_subsistenz, rns = ratio_nuss_subsistenz, rni = ratio_nuss_invest
    # Alternative: Lösung durch Bellman-Gleichung
    ge0 = lambda _list : all(_elem >= 0 for _elem in _list) # true, wenn alle Elemente in _list >= 0 sind
    in01 = lambda _list : all(0 <= _elem <= 1 for _elem in _list) # true, wenn alle Elemente in _list im Intervall [0,1] liegen
    # Gleichgewicht (Ressourcenrestriktion): Y_fisch = C_fisch_subsistenz + C_fisch_genuss
    Y_fisch = lambda _t, _x1 : A_fisch * ((1 + g_A)**_t) * (_x1**alpha_fisch) if ge0([_x1]) else np.nan
    C_fisch_subsistenz = lambda _t, _x1, _rfs : Y_fisch(_t, _x1) * _rfs       if ge0([_x1]) and in01([_rfs]) else np.nan
    C_fisch_genuss = lambda _t, _x1, _rfs : Y_fisch(_t, _x1) * (1 - _rfs)     if ge0([_x1]) and in01([_rfs]) else np.nan
    # Ohne Investition: Y_nuss = A_nuss * L_nuss^α_nuss, mit Investition: Y_nuss(t) = A_nuss (L_nuss * K_nuss(t))^α_nuss
    Y_nuss = lambda _t, _x2, _rns, _rni : A_nuss * ((1 + g_A)**_t) * ((_x2 * K_nuss(_t, _x2, _rns, _rni))**alpha_nuss) if ge0([_x2]) and in01([_rns,_rni]) else np.nan
    # Gleichgewicht (Ressourcenrestriktion): Y_nuss = C_nuss_subsistenz + C_nuss_genuss + C_nuss_invest
    C_nuss_subsistenz = lambda _t, _x2, _rns, _rni : Y_nuss(_t, _x2, _rns, _rni) * _rns                if ge0([_x2]) and in01([_rns,_rni]) else np.nan
    C_nuss_genuss = lambda _t, _x2, _rns, _rni : Y_nuss(_t, _x2, _rns, _rni) * (1 - _rns) * (1 - _rni) if ge0([_x2]) and in01([_rns,_rni]) else np.nan
    C_nuss_invest = lambda _t, _x2, _rns, _rni : Y_nuss(_t, _x2, _rns, _rni) * (1 - _rns) * _rni       if ge0([_x2]) and in01([_rns,_rni]) else np.nan
    # Investition I_nuss = A_nuss_invest * (C_nuss_invest**alpha_nuss_invest)
    I_nuss = lambda _t, _x2, _rns, _rni : A_nuss_invest * (C_nuss_invest(_t, _x2, _rns, _rni)**alpha_nuss_invest) if ge0([_x2]) and in01([_rns,_rni]) else np.nan
    K_nuss = lambda _t, _x2, _rns, _rni : K_nuss(_t - 1, _x2, _rns, _rni) + I_nuss(_t - 1, _x2, _rns, _rni) * (1 - K_nuss(_t - 1, _x2, _rns, _rni) / K_max) if _t > 0 else 1
    # Ressourcenrestriktion: Robinson kann in 1 Stunde Arbeit 10 Kokosnüsse einpflanzen
    _x5 = lambda _t, _x2, _rns, _rni : _x5_tmp if (_x5_tmp := C_nuss_invest(_t, _x2, _rns, _rni) / 10) >= 0 else np.nan # x5 = L_nuss_invest, C_nuss_invest/L_nuss_invest = 10
    # Zeitrestriktion: x4 = L_genuss = 24 - x1 - x2 - x3 - x5
    _x4 = lambda _t, _x1, _x2, _x3, _rns, _rni : _x4_tmp if (_x4_tmp := 24 - _x1 - _x2 - _x3 - _x5(_t, _x2, _rns, _rni)) >= 0 else np.nan
    # Nutzenfunktion U = U_subsistenz + U_genuss bleibt unverändert
    U_subsistenz = lambda _t, _x1, _x2, _x3, _rfs, _rns, _rni: A_subsistenz * ((MSR*C_fisch_subsistenz(_t, _x1, _rfs)+C_nuss_subsistenz(_t, _x2, _rns, _rni))**alpha_subsistenz) * (_x3**alpha_subsistenz) 
    U_genuss = lambda _t, _x1, _x2, _x3, _rfs, _rns, _rni : A_genuss * ((MSR*C_fisch_genuss(_t, _x1, _rfs) + C_nuss_genuss(_t, _x2, _rns, _rni))**alpha_genuss) * (_x4(_t, _x1, _x2, _x3, _rns, _rni)**alpha_genuss)
    # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
    U_delta = lambda _t, _x1, _x2, _x3, _rfs, _rns, _rni : (delta ** _t) * (U_subsistenz(_t, _x1, _x2, _x3, _rfs, _rns, _rni) + U_genuss(_t, _x1, _x2, _x3, _rfs, _rns, _rni))
    t_max = 5
    U_sum_tmax = lambda _x1, _x2, _x3, _rfs, _rns, _rni : sum(U_delta(_t, _x1, _x2, _x3, _rfs, _rns, _rni) for _t in range(t_max))
    U_sum = lambda _x1, _x2, _x3, _rfs, _rns, _rni : U_sum_tmp if np.isfinite(U_sum_tmp := U_sum_tmax(_x1, _x2, _x3, _rfs, _rns, _rni)) else -np.finfo(np.float32).max
    # Lösung durch Minimierung von -U_sum mit scipy.optimize.minimize mit Startwerten aus grid-search
    U_fun = lambda _x: -U_sum(_x[0], _x[1], _x[2], _x[3], _x[4], _x[5])
    initial_guess = [best_grid[0], best_grid[1], best_grid[2], best_grid[7], best_grid[8], best_grid[9]]
    sol_bounds = ((0, 24), (0, 24), (0, 24), (0, 1), (0, 1), (0, 1)) # Bounds für jede Variable: Sequence of (min, max) pairs, 0 ≤ L_i ​≤ 24, 0 ≤ r_i ​≤ 1
    print(f"  Gleichgewicht bei Subsistenz, Genusskonsum und Investition, Kapitalgrenze K_max={K_max}, Produktivitätssteigerung g_A={g_A*100}%, t_max={t_max}:")
    print(f"    U_sum(initial_guess) = {-U_fun(initial_guess):.1f}")
    sol_equilibriums = scipy.optimize.minimize(fun = U_fun, x0 = initial_guess, bounds=sol_bounds, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
    if sol_equilibriums.success:
        x1, x2, x3, rfs, rns, rni = sol_equilibriums.x # x1 = L_fisch, x2 = L_nuss, x3 = L_subsistenz, rfs = ratio_fisch_subsistenz, rns = ratio_nuss_subsistenz, rni = ratio_nuss_invest
        print(f"    L_fisch={x1:.1f}, L_nuss={x2:.1f}, L_subsistenz={x3:.1f}, ratio_fisch_subsistenz={rfs:.3f}, ratio_nuss_subsistenz={rns:.3f}, ratio_nuss_invest={rni:.3f}, U_sum={-sol_equilibriums.fun:.1f}")
    else:
        print(f"    ## WARNING: scipy.optimize.minimize failed")
        x1, x2, x3 = best_grid[0:3]     # x1 = L_fisch, x2 = L_nuss, x3 = L_subsistenz
        rfs, rns, rni = best_grid[7:10] # rfs = ratio_fisch_subsistenz, rns = ratio_nuss_subsistenz, rni = ratio_nuss_invest
    
    # Fall 3, Gleichgewicht bei Subsistenz, Genusskonsum und Investition in Kokosnussanbau: Plot Investition und Kapital über Zeit t
    t_values = np.arange(0, 300, 1)
    Y_fisch_values = []
    Y_nuss_values = []
    I_values = []
    K_values = []
    K_nuss = 1
    for t in t_values:
        Y_fisch_t = A_fisch * ((1 + g_A)**t) * (x1**alpha_fisch)
        Y_nuss_t = A_nuss * ((1 + g_A)**t) * ((x2 * K_nuss)**alpha_nuss)
        Y_fisch_values.append(Y_fisch_t)
        Y_nuss_values.append(Y_nuss_t)
        C_nuss_invest = Y_nuss_grid * (1-rns) * rni
        I_nuss = A_nuss_invest * (C_nuss_invest**alpha_nuss_invest)
        I_values.append(I_nuss)
        K_values.append(K_nuss)
        K_nuss = K_nuss + I_nuss * (1 - K_nuss / K_max)
    fig = plt.figure(figsize=(5, 5))
    plt.plot(t_values, Y_fisch_values, label=f"Y_fisch")
    plt.plot(t_values, Y_nuss_values, label=f"Y_nuss")
    plt.plot(t_values, I_values, label=f"Investition I")
    plt.plot(t_values, K_values, label=f"Kapital K")
    plt.xlabel("Zeit t")
    plt.yscale("log")
    plt.legend()
    if g_A <= 0:
        plt.title(f"Robinson-Crusoe-Wirtschaft:\nInvestition und Kapital über Zeit t\nKapitalgrenze K_max=100")
    else:
        plt.title(f"Robinson-Crusoe-Wirtschaft:\nInvestition und Kapital über Zeit t\nKapitalgrenze K_max=100, {g_A*100}% Produktivitätssteigerung")
    save_plot_png(pngfiles[0])
    print(f"")

# Lösung rcw_01_dynamics mit grid_search (Suche über alle Kombinationen von L_fisch, L_nuss, L_subsistenz, L_research, ratio_fisch_subsistenz, ratio_nuss_subsistenz, ratio_nuss_invest in einem Gitter mit gegebener Schrittweite
def rcw_01_dynamics_grid_search(fun, x_start, x_stop, x_steps, y_min):
    best_x = np.array(x_start, dtype=np.float32)
    best_y = y_min
    x_ranges = [np.arange(start, stop, step) for start, stop, step in zip(x_start, x_stop, x_steps)]
    for x in itertools.product(*x_ranges):
        y = -fun(x)
        if y > best_y:
            best_x = np.array(x, dtype=np.float32)
            best_y = y
            print(f"    grid_search: x={x}, y={y}")
    return best_x, best_y

# Lösung rcw_01_dynamics mit Iteration über grid_search
def rcw_01_dynamics_grid_search_iter(fun):
    grid_x_start = [0, 0, 0, 0, 0, 0, 0]
    grid_x_stop  = [24, 24, 24, 24, 1, 1, 1]
    grid_x_steps = [4, 4, 4, 4, 0.1, 0.1, 0.1]
    sol_grid_x = grid_x_start
    sol_grid_U = 0
    for grid_iter in range(3):
        grid_x, grid_U = rcw_01_dynamics_grid_search(fun, x_start=grid_x_start, x_stop=grid_x_stop, x_steps=grid_x_steps, y_min = sol_grid_U)
        if grid_U > sol_grid_U:
            sol_grid_x = grid_x
            sol_grid_U = grid_U
            print(f"    grid_search: sol_grid_x={sol_grid_x}, sol_grid_U={sol_grid_U}")
            for i in range(len(grid_x_start)):
                grid_x_start[i] = max(sol_grid_x[i] - grid_x_steps[i], grid_x_start[i])
                grid_x_stop[i]  = min(sol_grid_x[i] + grid_x_steps[i], grid_x_stop[i])
                grid_x_steps[i] *= 0.5
        else:
            break
    return sol_grid_x, sol_grid_U

# Plot Produktion, Investition, Kapital, Technologie, wirtschaftl. Kenngrößen und deren Wachstum
# in der dynamischen Wirtschaft mit den Werten aus rcw_01_dynamics
def plot_rcw_01_dynamics(values_t, plot_mode = 0):
    # Plot Produktion, Investition, Kapital und Technologie über Zeit t
    if plot_mode == 0 or plot_mode == 1:
        plt.plot(values_t["t"], values_t["Y_fisch"], label=f"Produktion Y_fisch")
        plt.plot(values_t["t"], values_t["Y_nuss"], label=f"Produktion Y_nuss")
        plt.plot(values_t["t"], values_t["I_nuss"], label=f"Investition I")
        plt.plot(values_t["t"], values_t["K"], label=f"Kapital K")
        plt.plot(values_t["t"], values_t["R_fn"], label=f"Technologie R")
        plt.legend()
        if plot_mode == 0:
            plt.title(values_t["title"])
            plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        if plot_mode == 1:
            plt.yscale("log")
            plt.ylim(bottom=1)
            plt.xlabel("Zeit t")
    # Plot wirtschaftlicher Kenngrößen über Zeit t
    elif plot_mode == 2:
        plt.plot(values_t["t"], values_t["BIP"], label=f"Produktion BIP")
        plt.plot(values_t["t"], values_t["K"], label=f"Kapital K")
        plt.plot(values_t["t"], values_t["I_nuss"], label=f"Investition I")
        plt.plot(values_t["t"], values_t["R_fn"], label=f"Technologie R")
        plt.plot(values_t["t"], values_t["AP"], label=f"Arbeitsproduktivität AP")
        plt.plot(values_t["t"], values_t["KP"], label=f"Kapitalproduktivität KP")
        plt.plot(values_t["t"], values_t["TFP"], label=f"Totale Faktorproduktivität TFP")
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        plt.ylim(bottom=-0.5)
        plt.legend()
        plt.title(values_t["title"])
    # Plot abs. Wachstum der wirtschaftlicher Kenngrößen über Zeit t
    elif plot_mode == 3:
        plt.plot(values_t["d_t"], values_t["d_BIP"], label=f"ΔBIP/Δt")
        plt.plot(values_t["d_t"], values_t["d_K"], label=f"ΔK/Δt")
        plt.plot(values_t["d_t"], values_t["d_I"], label=f"ΔI/Δt")
        plt.plot(values_t["d_t"], values_t["d_R"], label=f"ΔR/Δt")
        plt.plot(values_t["d_t"], values_t["d_AP"], label=f"ΔAP/Δt")
        plt.plot(values_t["d_t"], values_t["d_KP"], label=f"ΔKP/Δt")
        plt.plot(values_t["d_t"], values_t["d_TFP"], label=f"ΔTFP/Δt")
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
        plt.ylim(bottom=-0.5)
        plt.legend()
    # Plot rel. Wachstum der wirtschaftlicher Kenngrößen über Zeit t
    elif plot_mode == 4:
        plt.plot(values_t["g_t"], values_t["g_BIP"], label=f"ΔBIP/BIP(t)")
        plt.plot(values_t["g_t"], values_t["g_K"], label=f"ΔK/K(t)")
        plt.plot(values_t["g_t"], values_t["g_I"], label=f"ΔI/I(t)")
        plt.plot(values_t["g_t"], values_t["g_R"], label=f"ΔR/R(t)")
        plt.plot(values_t["g_t"], values_t["g_AP"], label=f"ΔAP/AP(t)")
        plt.plot(values_t["g_t"], values_t["g_KP"], label=f"ΔKP/KP(t)")
        plt.plot(values_t["g_t"], values_t["g_TFP"], label=f"ΔTFP/TFP(t)")
        plt.xlabel("Zeit t")
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        plt.ylim(bottom=-0.5)
        plt.legend()

# Robinson-Crusoe-Wirtschaft, 1-Personen-Wirtschaft:
# * Gleichgewicht zwischen Produktion und Konsum,
# * Dynamische Wirtschaft mit Subsistenzkonsum, Genusskonsum, Investition in Kokosnussanbau und Forschung,
# * Produktivitätssteigerung durch stetig verbesserte Technologie
# Wir modellieren Produktivitätssteigerungen durch das Wissenskapital R(t+1) = R(t) + A_research * (1 + R(t))^(α_research) * L_research
# mit dem Forschungsaufwand L_research rein als Arbeitsleistung.
# Der Parameter α_research ist entscheidend dafür, wie das Wissenskapital R(t) über der Zeit wächst:
# * α_research = 0 modelliert lineares Wachstum: R(t+1) = R(t) + A_research * L_research wächst linear, und damit wachsen auch die Produktivität A(R(t)) und die Produktion Y(t) ∝ A(R(t)) linear
# * α_research = 1 modelliert exponentielles Wachstum: R(t+1) = R(t) + A_research * (1 + R(t)) * L_research wächst exponentiell, und damit wachsen auch A(R(t)) und Y(t) exponentiell
# * 0 < α_research < 1 modelliert ein Wachstum zwischen linear und exponentiell
# Intertemporale Optimierung: die diskontierte Nutzenfunktion $U_sum = \sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
# t_max = 20 bedeutet: sum_{t=0}^{∞} wird ersetzt durch sum_{t=0}^{t_max} mit t_max = 20, da δ^t U(t) mit zunehmendem t immer kleiner wird (delta < 1). 
# Für t_max>=20 liefert scipy.optimize.minimize gleiche Lösungen (auf 3 Nachkommastellen genau); daher ist t_max=20 ist ein guter Kompromiss zwischen Genauigkeit und Rechenzeit.
def rcw_01_dynamics(alpha_research = 0, t_max = 20):

    # Shortcuts für alle Elemente in _list >= 0 bzw. im Intervall [0,1]
    ge0 = lambda _list : all(_elem >= 0 for _elem in _list) # true, wenn alle Elemente in _list >= 0 sind
    in01 = lambda _list : all(0 <= _elem <= 1 for _elem in _list) # true, wenn alle Elemente in _list im Intervall [0,1] liegen

    # Produzierte Mengen: Y_fisch = A_fisch * L_fisch ^ α_fisch, Y_nuss = A_nuss * L_nuss ^ α_nuss
    alpha_fisch = 0.5 # Elastizität 0.5: vierfacher Einsatz ergibt doppelte Menge Fisch
    alpha_nuss = 0.5  # Elastizität 0.5: vierfacher Einsatz ergibt doppelte Menge Nüsse

    # Investition in Kokosnüsse: I_nuss = A_nuss_invest * (C_nuss_invest * L_nuss_invest) ^ alpha_nuss_invest
    alpha_nuss_invest = 1.0 # Elastizität 1.0: doppelte Investition ergibt doppelte Menge
    A_nuss_invest = 0.5 # 1 Kokosnuss investiert ergibt 0.5 Kokosnüsse mehr pro Stunde Arbeit in der Zukunft (A_nuss_invest < 1, da Investition nicht 100% effizient, Ernteverluste, Unwetter, etc.)
    delta = 0.5 # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert.
    K_max = 100 # logistische Kapitalgrenze durch begrenzte Anbaufläche: K_nuss(t) = K_nuss(t-1) + I_nuss * (1 - K_nuss(t-1) / K_max)

    # Subsistenz-Nutzen: U_subsistenz = A_subsistenz * (MSR * C_fisch_subsistenz + C_nuss_subsistenz) ^ α_subsistenz * L_subsistenz ^ α_subsistenz
    MSR = 2 # Verhältnis der Präferenz für Fisch gegenüber Nüssen (marginale Substitutionsrate MSR): 1 Fisch entspricht 2 Nüssen im Nutzen
    A_subsistenz = 1 / ((MSR * 10 + 20)**2 * 10**2)
    alpha_subsistenz = 2 # Subsistenz mit höherer Elastizität

    # Genuss-Nutzen :U_genuss = A_genuss * (MSR * C_fisch_genuss + C_nuss_genuss) ^ α_genuss * L_genuss ^ α_genuss
    alpha_genuss = 0.2 # Niedrigere Elastizität für Genusskonsum (weniger wichtig als Subsistenzkonsum)
    U_subsistenz_10_20_11 = A_subsistenz * (MSR * 10 + 20)**2 * 11**2 # U_subsistenz mit 10 Fischen, 20 Kokosnüssen und 11 Stunden Freizeit
    A_genuss = U_subsistenz_10_20_11 / ((MSR * 3 + 4)**alpha_genuss * 5**alpha_genuss) # A_genuss aus U_subsistenz(10,20,11) = U_genuss(3,4,5) = U_genuss mit 3 Fischen, 4 Kokosnüssen und 5 Stunden Freizeit

    # Ressourcenrestriktion: Robinson kann in 1 Stunde Arbeit 10 Kokosnüsse einpflanzen
    ratio_C_nuss_invest_L_nuss_invest = 10 # C_nuss_invest/L_nuss_invest = 10
    
    # Zeitrestriktion: F = L_fisch + L_nuss + L_subsistenz + L_genuss + L_invest + L_research - 24 = 0
    L_sum = 24

    # Bisher: Produktivitätssteigerung durch Technologie: A(t) = A_0 (1 + g_A)^t, z.B. mit  g_A = 0.01, d.h. 1% Produktivitätssteigerung in jedem Zeitschritt
    # Jetzt: Produktivitätssteigerung als Folge von Arbeits- und Kapitaleinsatz in Forschung und Innovation:
    # In den Produktionfunktionen Y_i(t) = A_i(t)*(L_i(t)*K_i(t))^α_i werden die Niveaus A_i(t) jetzt abhängig von einem Wissenskapital (Technologielevel) R_i(t):
    # A_i(R_i(t)) = 1 + gamma_research R_i(t)
    # R_i(t+1) = R_i(t) + A_research L_i_research # Forschung in der Minimalvariante durch reinen Arbeitseinsatz, z.B. 1 Stunde Forschung erhöht das Technologielevel um 1, was in der nächsten Periode zu einer Produktivitätssteigerung von gamma_research führt
    gamma_research = 0.5 # 0.01 # Produktivitätssteigerung pro Einheit Wissenskapital: A_i(R_i(t)) = 1 + gamma_research * R_i(t), z.B. 1 Einheit Wissenskapital erhöht die Produktivität um 1%
    A_research = 1 # gleicher Forschungsproduktivitätsparameter A_research für Fisch und Nüsse, z.B. 1 Stunde Forschung erhöht das Technologielevel um 1, was in der nächsten Periode zu einer Produktivitätssteigerung von gamma_research führt
    A_fisch_base = 20 / (12**0.5) # Im Ausgangspunkt (ohne Forschung) produziert Robinson mit 12 Stunden Arbeit 20 Fische, also A_fisch = 20 / (12**0.5) für α_fisch = 0.5
    A_nuss_base = 30 / (12**0.5)  # Im Ausgangspunkt (ohne Forschung) produziert Robinson mit 12 Stunden Arbeit 30 Kokosnüsse, also A_nuss = 30 / (12**0.5) für α_nuss = 0.5
    A_fisch = lambda _Rf : A_fisch_base + gamma_research * _Rf # Produktivitätssteigerung durch Forschung: A_fisch(t) = A_fisch_base + gamma_research * R_fisch(t)
    A_nuss = lambda _Rn : A_nuss_base + gamma_research * _Rn   # Produktivitätssteigerung durch Forschung: A_nuss(t) = A_nuss_base + gamma_research * R_fisch(t)

    # Produktionsfunktion Fische: Y_fisch(t) = A_fisch * (L_fisch * K_fisch(t)) ^ α_fisch
    Y_fisch = lambda _x1, _Rf : A_fisch(_Rf) * (_x1**alpha_fisch) if ge0([_x1]) else np.nan

    # Gleichgewicht (Ressourcenrestriktion): Y_fisch = C_fisch_subsistenz + C_fisch_genuss
    C_fisch_subsistenz = lambda _x1, _rfs, _Rf : Y_fisch(_x1, _Rf) * _rfs   if ge0([_x1]) and in01([_rfs]) else np.nan
    C_fisch_genuss = lambda _x1, _rfs, _Rf : Y_fisch(_x1, _Rf) * (1 - _rfs) if ge0([_x1]) and in01([_rfs]) else np.nan

    # Produktionsfunktion Kokosnüsse: Y_nuss(t) = A_nuss * (L_nuss * K_nuss(t)) ^ α_nuss
    Y_nuss = lambda _x2, _Kn, _Rn : A_nuss(_Rn) * ((_x2 * _Kn)**alpha_nuss) if ge0([_x2]) else np.nan
    
    # Gleichgewicht (Ressourcenrestriktion): Y_nuss = C_nuss_subsistenz + C_nuss_genuss + C_nuss_invest
    C_nuss_subsistenz = lambda _x2, _rns, _Kn, _Rn : Y_nuss(_x2, _Kn, _Rn) * _rns                      if ge0([_x2]) and in01([_rns]) else np.nan
    C_nuss_genuss = lambda _x2, _rns, _rni, _Kn, _Rn : Y_nuss(_x2, _Kn, _Rn) * (1 - _rns) * (1 - _rni) if ge0([_x2]) and in01([_rns,_rni]) else np.nan
    C_nuss_invest = lambda _x2, _rns, _rni, _Kn, _Rn : Y_nuss(_x2, _Kn, _Rn) * (1 - _rns) * _rni       if ge0([_x2]) and in01([_rns,_rni]) else np.nan
    def C_nuss_invest_limited(_x1, _x2, _x3, _x6, _rns, _rni, _Kn, _Rn):
        _cni = C_nuss_invest(_x2, _rns, _rni, _Kn, _Rn)
        _x5 = _cni / ratio_C_nuss_invest_L_nuss_invest # x5 = L_nuss_invest = C_nuss_invest / 10
        _x4 = L_sum - _x1 - _x2 - _x3 - _x5 - _x6 # Zeitrestriktion: x4 = L_genuss = 24 - x1 - x2 - x3 - x5 - x6
        if _x4 >= 0:
            _x5 = L_sum - _x1 - _x2 - _x3 - _x4 - _x6 # Zeitrestriktion: x5 = L_nuss_invest = 24 - x1 - x2 - x3 - x4 - x6
            _cni = min(_cni, _x5 * ratio_C_nuss_invest_L_nuss_invest) # Ressourcenrestriktion: Robinson kann nicht mehr Kokosnüsse einpflanzen als Zeit zur Verfügung steht
        return _x4, _x5, _cni

    # Nutzenfunktion U = U_subsistenz + U_genuss bleibt unverändert
    U_subsistenz = lambda _x1, _x2, _x3, _rfs, _rns, _Kn, _Rf, _Rn : A_subsistenz * ((MSR*C_fisch_subsistenz(_x1, _rfs, _Rf) + C_nuss_subsistenz(_x2, _rns, _Kn, _Rn))**alpha_subsistenz) * (_x3**alpha_subsistenz) 
    U_genuss = lambda _x1, _x2, _x4, _rfs, _rns, _rni, _Kn, _Rf, _Rn : A_genuss * ((MSR*C_fisch_genuss(_x1, _rfs, _Rf) + C_nuss_genuss(_x2, _rns, _rni, _Kn, _Rn))**alpha_genuss) * (_x4**alpha_genuss)
    
    # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
    U_diskont_t = lambda _t, _x1, _x2, _x3, _x4, _rfs, _rns, _rni, _Kn, _Rf, _Rn : (delta ** _t) * (U_subsistenz(_x1, _x2, _x3, _rfs, _rns, _Kn, _Rf, _Rn) + U_genuss(_x1, _x2, _x4, _rfs, _rns, _rni, _Kn, _Rf, _Rn))

    # Investition I_nuss = A_nuss_invest * (C_nuss_invest**alpha_nuss_invest)
    I_nuss = lambda _cni : A_nuss_invest * (_cni**alpha_nuss_invest)

    # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{t_max}δ^t U(t)$ wird maximiert
    def U_sum(_x1, _x2, _x3, _x6, _rfs, _rns, _A_rni, _rni_t_0, _rni_t_inf):
        U_sum = 0
        K_nuss = 1 # Reales Kapital (Anzahl Kokosnüsse, Startwert)
        R_fn = 0   # Wissenskapital (Technologielevel) für Fisch- und Kokosnussproduktion, R_fn = R_fisch = R_nuss (Forschung bringt Produktivitätssteigerung in beiden Bereichen gleichermaßen)
        _rni = _rni_t_0 # Startwert für rni = ratio_nuss_invest 
        for _t in range(t_max):
            # Optimierung der diskontierten Nutzenfunktion
            _x4, _x5, _Cni = C_nuss_invest_limited(_x1, _x2, _x3, _x6, _rns, _rni, K_nuss, R_fn) # Investition in Kokosnüsse inkl. Ressourcenrestriktion
            if _x4 < 0: # Strafterm (negativer Nutzen) wenn Zeitrestriktion verletzt wird
                return -1e6 + _x4
            U_sum += U_diskont_t(_t, _x1, _x2, _x3, _x4, _rfs, _rns, _rni, K_nuss, R_fn, R_fn)
            K_nuss = K_nuss + I_nuss(_Cni) * (1 - K_nuss / K_max) # Investition I_nuss = A_nuss_invest * (C_nuss_invest**alpha_nuss_invest)
            R_fn = R_fn + A_research * ((1 + R_fn)**alpha_research) * _x6 # R(t+1) = R(t) + A_research * (1 + R(t))^(α_research) * L_research, Forschung durch reinen Arbeitseinsatz x6 = L_research, z.B. 1 Stunde Forschung erhöht das Technologielevel um 1, was in der nächsten Periode zu einer Produktivitätssteigerung von gamma_research führt
            _rni = _rni + _A_rni * (_rni_t_inf - _rni) # rni(t+1) = rni(t) + A_rni * (rni(t=oo) - rni(t)), Startwert rni(t=0) = _rni_t_0, Endwert rni(t=oo) = _rni_t_inf
        return U_sum
    
    # Gleichungssystem mit 9 Unbekannten: x1 = L_fisch, x2 = L_nuss, x3 = L_subsistenz, x6 = L_research, rfs = ratio_fisch_subsistenz, rns = ratio_nuss_subsistenz
    # rni(t+1) = rni(t) + A_rni * (rni_max - rni(t)) = ratio_nuss_invest, x[6]=A_rni, x[7]=rni(t=0), x[8]=rni(t=oo) 
    U_fun = lambda _x: -U_sum(_x[0], _x[1], _x[2], _x[3], _x[4], _x[5], _x[6], _x[7], _x[8])
    sol_bounds = ((0, L_sum), (0, L_sum), (0, L_sum), (0, L_sum), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)) # Bounds für jede Variable: Sequence of (min, max) pairs, 0 ≤ L_i ​≤ 24, 0 ≤ r_i ​≤ 1

    # Lösung mit grid_search (Suche über alle Kombinationen von L_fisch, L_nuss, L_subsistenz, L_research, ratio_fisch_subsistenz, ratio_nuss_subsistenz, ratio_nuss_invest in einem Gitter mit gegebener Schrittweite
    # Nicht optimiert, sehr langsam, nur als Plausibilitätscheck und für Startwerte für scipy.optimize.minimize, Lauf nur ohne Debugger empfehlenswert
    print(f"  Gleichgewicht bei Subsistenz, Genusskonsum, Investition und Produktivitätssteigerung durch Forschung, Kapitalgrenze K_max={K_max}, alpha_research={alpha_research}, grid-search:")
    # sol_grid_x, sol_grid_U = rcw_01_dynamics_grid_search_iter(fun = U_fun)
    sol_grid_x, sol_grid_U = [ 3.0, 5.0, 14.0, 0.0, 0.9499999, 0.775, 0.99999994 ], 6.175406944175536 # Ergebnis aus rcw_01_dynamics_grid_search_iter (dauert sehr lange...)
    print(f"    sol_grid_x={sol_grid_x}, sol_grid_U={sol_grid_U:.1f}")
    
    # Lösung durch Minimierung von -U_sum mit scipy.optimize.minimize mit Startwerten
    print(f"  Gleichgewicht bei Subsistenz, Genusskonsum, Investition und Produktivitätssteigerung durch Forschung, Kapitalgrenze K_max={K_max}, alpha_research={alpha_research}:")
    initial_guess = [ sol_grid_x[0], sol_grid_x[1], sol_grid_x[2], sol_grid_x[3], sol_grid_x[4], sol_grid_x[5], 0.1, 1.0, 0.0 ] # sol_grid_x = [ 3.0, 5.0, 14.0, 0.0, 0.9499999, 0.775, 0.99999994 ]
    sol_equilibriums = scipy.optimize.minimize(fun=U_fun, x0=initial_guess, bounds=sol_bounds, method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9})
    assert(sol_equilibriums.success)
    x1, x2, x3, x6, rfs, rns, A_rni, rni_t_0, rni_t_inf = sol_equilibriums.x # x1 = L_fisch, x2 = L_nuss, x3 = L_subsistenz, x6 = L_research, rfs = ratio_fisch_subsistenz, rns = ratio_nuss_subsistenz, A_rni, rni(t=0), rni(t=oo)
    print(f"    U_sum(initial_guess) = {-U_fun(initial_guess):.1f}, K_max={K_max}, alpha_research = {alpha_research}, t_max = {t_max}")
    print(f"    L_fisch={x1:.3f}, L_nuss={x2:.3f}, L_subsistenz={x3:.3f}, L_research={x6:.3f}, ratio_fisch_subsistenz={rfs:.3f}, ratio_nuss_subsistenz={rns:.3f}, A_rni={A_rni:.3f}, rni(t=0)={rni_t_0:.3f}, rni(t=oo)={rni_t_inf:.3f}, U_sum={-sol_equilibriums.fun:.1f}")
    # Beispiel (alpha_research=0, t_max=20): L_fisch=1.147, L_nuss=5.158, L_subsistenz=10.279, L_research=5.863, ratio_fisch_subsistenz=0.978, ratio_nuss_subsistenz=0.779, A_rni=0.232, rni(t=0)=1.000, rni(t=oo)=0.000, U_sum=11.2
    
    # Plot Produktion Y_fisch und Y_nuss, Investition I_nuss, Kapital K_nuss, Technologie R_fn sowie wirtschaftliche Kenngrößen und deren Wachstum über Zeit t
    values_t = {} # Alle Werte über Zeit t
    U_sum, K_nuss, R_fn, rni = 0, 1, 0, rni_t_0 # Startwerte
    for t in np.arange(0, 100, 1):
        # Berechnung der diskontierten Nutzenfunktion
        x4, x5, C_ni = C_nuss_invest_limited(x1, x2, x3, x6, rns, rni, K_nuss, R_fn)
        if x4 < 0: # Zeitrestriktion verletzt
            break
        U_sum += U_diskont_t(t, x1, x2, x3, x4, rfs, rns, rni, K_nuss, R_fn, R_fn)
        # Berechnung wirtschaftl. Kenngrößen:
        val = {"t":t, "L_fisch":x1, "L_nuss":x2, "L_subsistenz":x3, "L_genuss":x4, "L_nuss_invest":x5, "L_research":x6, "rfs":rfs, "rns":rns, "rni":rni, "K_nuss":K_nuss, "R_fn":R_fn, "U_sum":U_sum }
        val["Y_fisch"] = Y_fisch(x1, R_fn)               # Produktion von Fischen
        val["Y_nuss"] = Y_nuss(x2, K_nuss, R_fn)         # Produktion von Kokosnüssen
        val["I_nuss"] = I_nuss(C_ni)                     # Investition in Kokosnüsse
        val["K"] = K_nuss                                # Kapital: K(t) = K_fisch(t) + K_nuss(t) mit K_fisch(t) = 0, da nur in Kokosnüsse investiert wird
        val["BIP"] = MSR * val["Y_fisch"] + val["Y_nuss"]  # Reale Wirtschaftsleistung: BIP(t) = 2 * Y_fisch(t) + Y_nuss(t)
        val["AP"] = val["BIP"] / (x1 + x2 + x5)          # Arbeitsproduktivität (Verhältnis zwischen BIP und Arbeitssumme L): AP(t) =  = BIP(t) / (L_fisch(t) + L_nuss(t) + L_invest(t))
        val["KP"] = val["BIP"] / val["K"]                # Kapitalproduktivität (Verhältnis zwischen BIP und Kapital K): KP(t) = BIP(t) / K(t)
        val["TFP"] = val["BIP"] / (MSR * (x1**alpha_fisch) + (x2 * K_nuss)**alpha_nuss)       # Totale Faktorproduktivität: TFP(t) = BIP(t) / (2 * (L_fisch * K_fisch(t))^α_fisch + (L_nuss * K_nuss(t))^α_nuss)
        if t > 0:
            val["g_t"] = t
            val["g_BIP"] = (val["BIP"] - values_t["BIP"][-1]) / values_t["BIP"][-1]         # Rel. Wirtschaftswachstum: g_BIP(t) = (BIP(t) - BIP(t-1)) / BIP(t-1)
            val["g_K"] = (val["K"] - values_t["K"][-1]) / values_t["K"][-1]                 # Rel. Kapitalwachstum: g_K(t) = (K(t) - K(t-1)) / K(t - 1)
            val["g_I"] = (val["I_nuss"] - values_t["I_nuss"][-1]) / values_t["I_nuss"][-1] if values_t["I_nuss"][-1] != 0 else np.nan  # Rel. Investitionswachstum: g_I(t) = (I(t) - I(t-1)) / I(t - 1)
            val["g_R"] = (val["R_fn"] - values_t["R_fn"][-1]) / values_t["R_fn"][-1] if values_t["R_fn"][-1] != 0 else np.nan          # Rel. Technologiewachstum: g_R(t) = (K(t) - R(t-1)) / R(t - 1)
            val["g_AP"] = (val["AP"] - values_t["AP"][-1]) / values_t["AP"][-1]             # Rel. Arbeitsproduktivitätswachstum: g_AP(t) = (AP(t) - AP(t-1)) / AP(t-1) 
            val["g_KP"] = (val["KP"] - values_t["KP"][-1]) / values_t["KP"][-1]             # Rel. Kapitalproduktivitätswachstum: g_KP(t) = (KP(t) - KP(t-1)) / KP(t-1) 
            val["g_TFP"] = (val["TFP"] - values_t["TFP"][-1]) / values_t["TFP"][-1]         # Rel. TFP-Wachstum: g_TFP(t) = (TFP(t) - TFP(t-1)) / TFP(t-1) 
            dt = (t - values_t["t"][-1])
            val["d_t"] = t
            val["d_BIP"] = (val["BIP"] - values_t["BIP"][-1]) / dt                          # Abs. Wirtschaftswachstum: ΔBIP / Δt
            val["d_K"] = (val["K"] - values_t["K"][-1]) / dt                                # Abs. Kapitalwachstum: ΔK / Δt
            val["d_I"] = (val["I_nuss"] - values_t["I_nuss"][-1]) / dt                      # Abs. Investitionswachstum: ΔI / Δt
            val["d_R"] = (val["R_fn"] - values_t["R_fn"][-1]) / dt                          # Abs. Technologiewachstum: ΔR / Δt
            val["d_AP"] = (val["AP"] - values_t["AP"][-1]) / dt                             # Abs. Arbeitsproduktivitätswachstum: ΔAP / Δt
            val["d_KP"] = (val["KP"] - values_t["KP"][-1]) / dt                             # Abs. Kapitalproduktivitätswachstum: ΔKP / Δt
            val["d_TFP"] = (val["TFP"] - values_t["TFP"][-1]) / dt                          # Abs. TFP-Wachstum: ΔTFP / Δt 
        # Append aller Kenngrößen für alle plots
        for key in val.keys():
            if key not in values_t:
                values_t[key] = []
            values_t[key].append(val[key])
        # Rekursion für Kapital K_nuss, Technologie R_fn und Investitionsanteil rni
        K_nuss = K_nuss + I_nuss(C_ni) * (1 - K_nuss / K_max)
        R_fn = R_fn + A_research * ((1 + R_fn)**alpha_research) * x6
        rni = rni + A_rni * (rni_t_inf - rni)
    values_t["title"] = f"Kapitalgrenze K_max={K_max}, α_research={alpha_research}"
    print(f"")
    return values_t

if __name__ == "__main__":

    # Plot Transformationskurve und Indifferenzkurve für Fische und Kokosnüsse
    rcw_01_plot_ppc() # rcw_01_plot_ppc(pngfiles=["mini_wm_rcw_01_ppc.png", "mini_wm_rcw_01_ind.png"])
    
    # Gleichgewicht zwischen Produktion und Konsum in 3 Fällen: 1. Reine Subsistenz, 2. Subsistenz und Genusskonsum, 3. Subsistenz, Genusskonsum und Investition
    rcw_01_equilibrium() # rcw_01_equilibrium(pngfiles=["mini_wm_rcw_01_invest.png"])

    # Dynamische Wirtschaft mit Subsistenzkonsum, Genusskonsum, Investition in Kokosnussanbau und Produktivitätssteigerung durch Forschung und verbesserte Technologie
    rcw_01_values1 = rcw_01_dynamics(alpha_research = -10, t_max = 20)
    rcw_01_values2 = rcw_01_dynamics(alpha_research = 0.0, t_max = 20)
    rcw_01_values3 = rcw_01_dynamics(alpha_research = 0.5, t_max =  5)

    # Plot Produktion, Investition, Kapital und Technologie in der dynamischen Wirtschaft
    fig, _ = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Robinson-Crusoe-Wirtschaft: Produktion, Investition, Kapital und Technologie über Zeit")
    plt.subplot(2, 3, 1)
    plot_rcw_01_dynamics(rcw_01_values1, plot_mode = 0)
    plt.subplot(2, 3, 2)
    plot_rcw_01_dynamics(rcw_01_values2, plot_mode = 0)
    plt.subplot(2, 3, 3)
    plot_rcw_01_dynamics(rcw_01_values3, plot_mode = 0)
    plt.subplot(2, 3, 4)
    plot_rcw_01_dynamics(rcw_01_values1, plot_mode = 1)
    plt.subplot(2, 3, 5)
    plot_rcw_01_dynamics(rcw_01_values2, plot_mode = 1)
    plt.subplot(2, 3, 6)
    plot_rcw_01_dynamics(rcw_01_values3, plot_mode = 1)
    # save_plot_png("mini_wm_rcw_01_invest3a.png")

    # Plot wirtschaftl. Kenngrößen und deren Wachstum in der dynamischen Wirtschaft
    fig, _ = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(f"Robinson-Crusoe-Wirtschaft: Wirtschaftl. Kenngrößen und deren Wachstum über Zeit")
    plt.subplot(3, 3, 1)
    plot_rcw_01_dynamics(rcw_01_values1, plot_mode = 2)
    plt.subplot(3, 3, 2)
    plot_rcw_01_dynamics(rcw_01_values2, plot_mode = 2)
    plt.subplot(3, 3, 3)
    plot_rcw_01_dynamics(rcw_01_values3, plot_mode = 2)
    plt.subplot(3, 3, 4)
    plot_rcw_01_dynamics(rcw_01_values1, plot_mode = 3)
    plt.subplot(3, 3, 5)
    plot_rcw_01_dynamics(rcw_01_values2, plot_mode = 3)
    plt.subplot(3, 3, 6)
    plot_rcw_01_dynamics(rcw_01_values3, plot_mode = 3)
    plt.subplot(3, 3, 7)
    plot_rcw_01_dynamics(rcw_01_values1, plot_mode = 4)
    plt.subplot(3, 3, 8)
    plot_rcw_01_dynamics(rcw_01_values2, plot_mode = 4)
    plt.subplot(3, 3, 9)
    plot_rcw_01_dynamics(rcw_01_values3, plot_mode = 4)
    # save_plot_png("mini_wm_rcw_01_invest3b.png")

    plt.show()
    print(f"")
