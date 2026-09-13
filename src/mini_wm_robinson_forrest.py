"""
Mini-WM: Ein minimalistisches Wirtschaftsmodell,
Teil 2: Robinson-Crusoe-Wirtschaft mit einem öffentlichem Gut: 
dem Inselwald, den Robinson und Freitag zur Brennholzproduktion benutzen.
Erläuterungen siehe MiniWM_Teil02_RobinsonFreitag.md.
"""
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy # pip install scipy
import scipy.optimize
import sympy as sp # pip install sympy
from mini_wm_robinson_two_persons import fmt, GridStepType, rcw_02_grid_search_iter, save_plot_png

# Formatierte Ausgabe der Spielmatrix mit Nutzenpaaren (U1, U2) für Robinson und Freitag
def print_game_matrix(game_matrix, fmtstr="5.2f", col_prefix = ""):
    fmt = lambda x: f"{x:{fmtstr}}"
    for i, row in enumerate(game_matrix):
        U_str = "  ".join([f"({fmt(U1)}, {fmt(U2)})" for U1, U2 in game_matrix[i]])
        print(f"{col_prefix}{U_str}")

# Suche nach stark oder schwach dominanten Strategien für beide Spieler in einer gegebenen Spielmatrix. 
# Zurückgeben wird eine Liste von dominanten Strategien für jeden Spieler:
# dominant_strategies[0] = Liste der dominanten Strategien von Robinson (Zeilenspieler), 
# dominant_strategies[1] = Liste der dominanten Strategien von Freitag (Spaltenspieler)
def find_dominant_strategies(game_matrix):
    num_rows, num_cols = game_matrix.shape
    dominant_strategies = [[], []] # Listen für dominante Strategien von Robinson (Spieler 1, Zeilenspieler) und Freitag (Spieler 2, Spaltenspieler)
    # Dominante Strategien für Spieler 1
    for i in range(num_rows):
        is_dominant = True
        for j in range(num_rows):
            is_dominant = all(game_matrix[i, k][0] >= game_matrix[j, k][0] for k in range(num_cols)) # Vergleich U1[i, k] >= U1[j, k] in allen Spalten k, sonst ist Strategie i nicht dominant
            if not is_dominant:
                break
        if is_dominant:
            dominant_strategies[0].append(i)
    # Dominante Strategien für Spieler 2
    for k in range(num_cols):
        is_dominant = True
        for l in range(num_cols):
            is_dominant = all(game_matrix[i, k][1] >= game_matrix[i, l][1] for i in range(num_rows)) # Vergleich U2[i, k] >= U2[i, l] in allen Zeilen i, sonst ist Strategie k nicht dominant)
            if not is_dominant:
                break
        if is_dominant:
            dominant_strategies[1].append(k)
    return dominant_strategies

# Prüft ob ein Nash-Gleichgewicht Pareto-optimal ist. Ein Nash-Gleichgewicht ist Pareto-optimal, wenn es keine andere Strategiekombination gibt, die beide Spieler besser stellt.
def equilibrium_is_pareto_optimal(game_matrix, equilibrium):
    i_best, j_best = equilibrium[0], equilibrium[1]
    U1_best, U2_best = game_matrix[i_best, j_best]
    is_pareto_optimal = True
    num_rows, num_cols = game_matrix.shape
    for i in range(num_rows):
        for j in range(num_cols):
            U1_ij, U2_ij = game_matrix[i, j]
            if (U1_ij > U1_best and U2_ij >= U2_best) or (U1_ij >= U1_best and U2_ij > U2_best):
                # Es gibt eine andere Strategiekombination (i, j), die beide Spieler besser stellt
                i_best, j_best = i, j
                U1_best, U2_best = game_matrix[i_best, j_best]
                is_pareto_optimal = False 
    return is_pareto_optimal, (i_best, j_best), (U1_best, U2_best)

# Nash-Gleichgewichte am Beispiel Holzproduktion und gemeinschaftlichem Holzkonsum (Lagerfeuer):
# Brennholzproduktion: Y1_holz = A1_holz * L1_holz, Y2_holz = A2_holz * L2_holz
# mit den Faktorproduktivitäten A1_holz = 1 und A2_holz = 0.5
# Holzproduktion ist gleich Holzkonsum: C1_holz = Y1_holz, C2_holz = Y2_holz
# Nutzen U1_holz, U2_holz des gemeinsamen Lagerfeuers:
# U1_holz = U2_holz = A_warme * (A1_holz * L1_holz + A2_holz * L2_holz)
# Alternativnutzen, wenn der Arbeitseinsatz Li_holz anderweitig verwendet wird:
# U1_alternativ = A1_alternativ * L1_holz, U2_alternativ = A2_alternativ * L2_holz
def rcw_02_wood_production():

    A1_holz = 1   # Robinsons Faktorproduktivität in der Holzproduktion
    A2_holz = 0.5 # Freitags Faktorproduktivität in der Holzproduktion
    A1_alternativ = 1 # Robinsons Faktorproduktivität in der Alternativproduktion
    A2_alternativ = 1 # Freitags Faktorproduktivität in der Alternativproduktion
    alpha_holz = 1 # Produktionselastizität der Holzproduktion, Yi_holz = Ai_holz * (Li_holz^alpha_holz)
    alpha_alternativ = 1 # Produktionselastizität der Alternativproduktion, Yi_alternativ = Ai_alternativ * (Li_alternativ^alpha_alternativ)
    Li_range = range(0, 3) # Robinson bzw. Freitag arbeiten je 0, 1 oder 2 Stunden in der Holzproduktion (Li_holz) oder in der alternativen Tätigkeit (Li_alternativ = 2 - Li_holz)
    print(f"Spielmatrizen für Holzproduktion und gemeinschaftlichen Holzkonsum mit 2 Spielern (i=1,2)")
    print(f"Einsatz: Li_holz = [0, 1, 2], Li_alternativ = (2 - Li_holz)")
    print(f"Holz-Produktion: Yi_holz = Ai_holz * Li_holz")
    print(f"Holz-Nutzen autark, ohne Tauschhandel: Ci_holz = Yi_holz, Ui_holz = A_warme * (Ai_holz * Li_holz)")
    print(f"Holz-Nutzen gemeinsam, ohne Tauschhandel: Ci_holz = Y1_holz + Y2_holz, Ui_holz = A_warme * (A1_holz * L1_holz^alpha_holz + A2_holz * L2_holz^alpha_holz)")
    print(f"Alternativ-Produktion: Yi_alternativ = Ai_alternativ * Li_alternativ")
    print(f"Alternativ-Nutzen, ohne Tauschhandel: Ci_alternativ = Yi_alternativ, Ui_alternativ = Ai_alternativ * Li_alternativ^alpha_alternativ")
    print(f"Nutzen: Ui = Ui_holz + Ui_alternativ")
    print(f"Beispielparameter: A1_holz = {A1_holz}, A2_holz = {A2_holz}, alpha_holz={alpha_holz}, A1_alternativ = {A1_alternativ}, A2_alternativ = {A2_alternativ}, alpha_alternativ={alpha_alternativ}")
    
    plt_fig, plt_idx = None, 0
    for A_warme in [0.5, 0.9, 4]: # Nutzenfaktor für die Wärme des Lagerfeuers: Ui_holz = A_warme * (C1_holz + C2_holz)

        # Alternativarbeit = Gesamtzeit (2 Stunden) minus Arbeit in der Holzproduktion
        Li_max = Li_range.stop - 1 # Maximal 2 Stunden Arbeit
        Li_alt = lambda Li_holz: Li_max - Li_holz

        # Spielmatrix mit Nutzenfunktion Ui = Ui_holz + Ui_alt(2-Li_holz) für alle Kombinationen von L1_holz und L2_holz in {0, 1, 2}
        U1_holz = lambda L1_holz, L2_holz: A_warme * (A1_holz * L1_holz**alpha_holz + A2_holz * L2_holz**alpha_holz) # Robinsons Nutzen aus der Holzproduktion (gemeinsames Lagerfeuer)
        U2_holz = lambda L1_holz, L2_holz: A_warme * (A1_holz * L1_holz**alpha_holz + A2_holz * L2_holz**alpha_holz) # Freitags Nutzen aus der Holzproduktion (gemeinsames Lagerfeuer)
        U1_alt = lambda L1_holz: A1_alternativ * Li_alt(L1_holz)**alpha_alternativ # Robinsons Alternativnutzen
        U2_alt = lambda L2_holz: A2_alternativ * Li_alt(L2_holz)**alpha_alternativ # Freitags Alternativnutzen
        game_matrix = np.zeros((Li_range.stop, Li_range.stop), dtype=object)
        for L1_holz in Li_range:
            for L2_holz in Li_range:
                game_matrix[L1_holz, L2_holz] = (U1_holz(L1_holz, L2_holz) + U1_alt(L1_holz), U2_holz(L1_holz, L2_holz) + U2_alt(L2_holz))
        # Dominante Strategien suchen, Nash-Gleichgewichte auf Pareto-Optimalität prüfen
        dominant_strategies = find_dominant_strategies(game_matrix)
        print(f"\n  A_warme = {A_warme:.1f}, L1_holz = {Li_range}, L2_holz = {Li_range}")
        print_game_matrix(game_matrix, "5.2f", "    ")
        for s1_idx in dominant_strategies[0]: # Dominante Strategien für Robinson (Zeilenspieler)
            for s2_idx in dominant_strategies[1]: # Dominante Strategien für Freitag (Spaltenspieler)
                # Es existiert eine dominante Strategie für beide Spieler => Ist das Nash-Gleichgewicht Pareto-optimal?
                L1_holz_dominant, L2_holz_dominant = Li_range[s1_idx], Li_range[s2_idx]
                U1_dominant, U2_dominant = game_matrix[L1_holz_dominant, L2_holz_dominant]
                is_pareto_optimal, S_pareto, U_pareto = equilibrium_is_pareto_optimal(game_matrix, (L1_holz_dominant, L2_holz_dominant))
                print(f"  Dominante Strategien: L1_holz = {L1_holz_dominant}, L2_holz = {L2_holz_dominant}, U = ({U1_dominant}, {U2_dominant})")
                if is_pareto_optimal:
                    print(f"  Das Nash-Gleichgewicht L_holz = {S_pareto} mit Nutzen U = {U_pareto} ist Pareto-optimal")
                else:
                    print(f"  Das Nash-Gleichgewicht L_holz = ({L1_holz_dominant}, {L2_holz_dominant}) mit Nutzen U = ({U1_dominant}, {U2_dominant}) ist nicht Pareto-optimal: Strategien Li_holz = {S_pareto} mit Nutzen {U_pareto} stellt beide besser")

        # Autarkie-Nutzen: Robinson und Freitag produzieren und konsumieren jeweils nur für sich selbst. Im ersten Schritt sei der Wald eine unbegrenzte Ressource.
        U1_autarkie_fun = lambda x: A_warme * (A1_holz * x**alpha_holz) + A1_alternativ * Li_alt(x)**alpha_alternativ # 1 Unbekannte: x = L1_holz
        U2_autarkie_fun = lambda x: A_warme * (A2_holz * x**alpha_holz) + A2_alternativ * Li_alt(x)**alpha_alternativ # 1 Unbekannte: x = L2_holz
        sol1_autarkie = scipy.optimize.minimize(fun = lambda x: -U1_autarkie_fun(x), x0 = [0], bounds=((0, Li_max),), method="SLSQP", options={"maxiter": 1000, "ftol": 1e-6})
        sol2_autarkie = scipy.optimize.minimize(fun = lambda x: -U2_autarkie_fun(x), x0 = [0], bounds=((0, Li_max),), method="SLSQP", options={"maxiter": 1000, "ftol": 1e-6})
        if sol1_autarkie.success and sol2_autarkie.success:
            L1_holz_autarkie = sol1_autarkie.x[0]
            L2_holz_autarkie = sol2_autarkie.x[0]
            print(f"  Autarkie bei unbegrenztem Wald: L1_holz_autarkie={L1_holz_autarkie:.2f}, L2_holz_autarkie={L2_holz_autarkie:.2f}, U1_autarkie={U1_autarkie_fun(L1_holz_autarkie):.2f}, U2_autarkie={U2_autarkie_fun(L2_holz_autarkie):.2f}")
        else:
            print(f"  Autarkie bei unbegrenztem Wald: scipy.optimize.minimize failed (\"{sol1_autarkie.message}\", \"{sol2_autarkie.message}\")")
            continue

        # Tauschhandel und gemeinsames Lagerfeuer:
        Y1_holz = lambda L1_holz: A1_holz * L1_holz**alpha_holz # Robinsons Holzproduktion
        Y2_holz = lambda L2_holz: A2_holz * L2_holz**alpha_holz # Freitags Holzproduktion
        Y1_alternativ = lambda L1_holz: A1_alternativ * Li_alt(L1_holz)**alpha_alternativ # Robinsons Alternativproduktion
        Y2_alternativ = lambda L2_holz: A2_alternativ * Li_alt(L2_holz)**alpha_alternativ # Freitags Alternativproduktion
        # Durch seine höhere Holzproduktion kann Robinson mehr vom öffentlichen Gut Wärme produzieren, dafür kompensiert Freitag ihn mit tau2_alternativ*Y2_alternativ seiner Alternativgüter. 
        # Mit den unterschiedlichen Faktorproduktivitäten der Holzproduktion profitieren beide davon.
        C1_holz = lambda L1_holz: Y1_holz(L1_holz) # Robinsons Brennholzkonsum = Robinsons Holzproduktion
        C2_holz = lambda L2_holz: Y2_holz(L2_holz) # Freitags Brennholzkonsum = Freitags Holzproduktion
        C1_alternativ = lambda L1_holz, L2_holz, tau2_alternativ: Y1_alternativ(L1_holz) + tau2_alternativ * Y2_alternativ(L2_holz) # Robinsons Konsum von Alternativprodukten nach Tauschhandel
        C2_alternativ = lambda L1_holz, L2_holz, tau2_alternativ: Y2_alternativ(L2_holz) - tau2_alternativ * Y2_alternativ(L2_holz) # Freitags Konsum von Alternativprodukten nach Tauschhandel
        # Nutzen des gemeinsamen Lagerfeuers nach Tauschhandel: Ui = A_warme * (C1_holz + C2_holz) + Ci_alternativ
        U1_sum_fun = lambda L1_holz, L2_holz, tau2_alternativ: A_warme * (C1_holz(L1_holz) + C2_holz(L2_holz)) + C1_alternativ(L1_holz, L2_holz, tau2_alternativ)
        U2_sum_fun = lambda L1_holz, L2_holz, tau2_alternativ: A_warme * (C1_holz(L1_holz) + C2_holz(L2_holz)) + C2_alternativ(L1_holz, L2_holz, tau2_alternativ)

        # Tauschhandel und gemeinsames Lagerfeuer mit unbegrenztem Wald: 
        # 3 Unbekannte: x[0] = L1_holz, x[1] = L2_holz, x[2] = tau2_alternativ
        U1_x_fun = lambda x: U1_sum_fun(x[0], x[1], x[2])
        U2_x_fun = lambda x: U2_sum_fun(x[0], x[1], x[2])
        # Nash-Produkt der Nutzenfunktionen, Maximierung führt zu Pareto-effizienter Lösung mit positivem Handelsgewinn für beide
        U1_autarkie = U1_x_fun([L1_holz_autarkie, L2_holz_autarkie, 0])
        U2_autarkie = U2_x_fun([L1_holz_autarkie, L2_holz_autarkie, 0])
        nash_product = lambda x: (U1_x_fun(x) - U1_autarkie) * (U2_x_fun(x) - U2_autarkie)
        sol_equilibrium = scipy.optimize.minimize(fun = lambda x: -nash_product(x), 
            x0 = [L1_holz_autarkie, L2_holz_autarkie, 0.1], bounds=((0, Li_max), (0, Li_max), (0, 1)), method="SLSQP", options={"maxiter": 1000, "ftol": 1e-9},
            constraints=[{"type": "ineq", "fun": lambda x: U1_x_fun(x) - U1_autarkie}, {"type": "ineq", "fun": lambda x: U2_x_fun(x) - U2_autarkie}]) # Randbedingungen: U1_x(x) >= U1_autarkie, U2_x(x) >= U2_autarkie
        if sol_equilibrium.success:
            sol_x = sol_equilibrium.x
            print(f"  Tauschhandel bei unbegrenztem Wald: L1_holz={sol_x[0]:.2f}, L2_holz={sol_x[1]:.2f}, tau2_alternativ={sol_x[2]:.2f}, U1={U1_x_fun(sol_x):.2f}, U2={U2_x_fun(sol_x):.2f}")
        else:
            print(f"  Tauschhandel bei unbegrenztem Wald: scipy.optimize.minimize failed (\"{sol_equilibrium.message}\")")

        # Der Wald ist eine begrenzte Ressource, Modellierung als begrenztes Kapital:
        # K_wald_left(t) = K_wald(t) - Y1_holz(t) - Y2_holz(t)
        # K_wald(t+1) = K_wald_left(t) + (g_wald * K_wald_left(t)) * (1 - K_wald_left(t) / K_wald_max)
        K_wald_max = 100 # Maximale Größe des Waldes, z.B. 100 Bäume
        g_wald = 0.05    # Natürliche Wachstumsrate des Waldes, z.B. 5% pro Zeiteinheit
        t_max = 100      # Maximale Anzahl von Zeiteinheiten, um die intertemporale Optimierung durchzuführen
        
        # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
        # x[0] = L1_holz, x[1] = L2_holz, x[2] = tau2_alternativ, 
        # delta1, delta2 = Diskontfaktoren, 
        # penalty_fun(x) = Straffunktion (Regulierung) oder None
        # U1_diskont += (delta1**t) * (U1(x) - penalty_fun(x)[0])
        # U2_diskont += (delta2**t) * (U2(x) - penalty_fun(x)[1])
        def U_diskont_fun(x, delta1, delta2, penalty_fun = None, autarkie = False):
            U1_diskont = 0
            U2_diskont = 0
            K_wald = K_wald_max
            values = {"t": [], "L1_holz": [], "L2_holz": [], "Y1_holz": [], "Y2_holz": [], "U1": [], "U2": [], "K": [] }
            t_wald_exploited = t_max + 1
            for t in range(t_max):
                L1_holz, L2_holz, tau2_alternativ = x
                # Der Wald wird als begrenztes Kapital modelliert
                K_wald_left = K_wald - Y1_holz(L1_holz) - Y2_holz(L2_holz) # Waldbestand nach Abholzung mit Y1_holz+Y2_holz
                if K_wald_left >= 0:
                    K_wald = K_wald_left + (g_wald * K_wald_left) * (1 - K_wald_left / K_wald_max)
                else: # Randbedingung: Robinson und Freitag können nicht mehr Holz fällen als Bäume existieren:
                    if K_wald > 0:
                        # Aufteilung des Restwaldes K_wald: Y1_holz(L1_holz_left) + Y2_holz(L2_holz_left) = K_wald
                        factor_rest = A1_holz * L1_holz + A2_holz * L2_holz
                        L1_holz_rest = K_wald * A1_holz * L1_holz / factor_rest if factor_rest > 1.0e-6 else 0 # Robinsons Anteil am restlich verbliebenen Wald
                        L2_holz_rest = K_wald * A2_holz * L2_holz / factor_rest if factor_rest > 1.0e-6 else 0 # Freitags Anteil am restlich verbliebenen Wald
                        L1_holz = L1_holz_rest
                        L2_holz = L2_holz_rest
                    else:
                        L1_holz, L2_holz = 0, 0 # Ohne Wald keine Holzproduktion => Li_holz=0, Yi_holz=0, K_wald<=0
                        tau2_alternativ = 0     # Ohne Holz kein Tausch Alternativprodukt gegen Wärme aus Holz
                    K_wald = 0
                    t_wald_exploited = min(t, t_wald_exploited)
                # Intertemporale Optimierung: die diskontierte Nutzenfunktion $\sum_{t=0}^{∞}δ^t U(t)$ wird maximiert
                penalties = penalty_fun([L1_holz, L2_holz, tau2_alternativ]) if penalty_fun is not None else [0, 0]
                if autarkie:
                    U1_t = U1_autarkie_fun(L1_holz) - penalties[0]
                    U2_t = U2_autarkie_fun(L2_holz) - penalties[1]
                else:
                    U1_t = U1_sum_fun(L1_holz, L2_holz, tau2_alternativ) - penalties[0]
                    U2_t = U2_sum_fun(L1_holz, L2_holz, tau2_alternativ) - penalties[1]
                U1_diskont += (delta1**t) * U1_t
                U2_diskont += (delta2**t) * U2_t
                # Ausgabe Li, Ui, K_wald über t für plots
                values["t"].append(t)
                values["L1_holz"].append(L1_holz)
                values["L2_holz"].append(L2_holz)
                values["Y1_holz"].append(Y1_holz(L1_holz))
                values["Y2_holz"].append(Y2_holz(L2_holz))
                values["U1"].append(U1_t)
                values["U2"].append(U2_t)
                values["K"].append(K_wald)
            t_wald_exploited = f">{t_max}" if t_wald_exploited > t_max else f"={t_wald_exploited}"
            return U1_diskont, U2_diskont, values, K_wald, t_wald_exploited

        # Autarkie-Nutzen: Robinson und Freitag produzieren und konsumieren jeweils nur für sich selbst.
        # Jeder maximiert seinen Nutzen Ui = Ui_holz + Ui_alt(2-Li_holz) = A_warme * (Ai_holz * Li_holz) + Ai_alternativ * Li_alternativ mit 0 <= Li_holz <= 2 und Li_alternativ = 2 - Li_holz
        # Der Drohpunkt (d1,d2) des Nashproduktes (U1-d1)*(U2-d2) ist der Autarkie-Nutzen, d.h. der Nutzen den beide Spieler bei Autarkie erreichen, wenn Verhandlungen scheitern (keine Kooperation, kein Tauschhandel, jeder produziert nur für sich).
        # Anmerkung: Robinson und Freitag haben zwei unterschiedliche Verhandlungen und Drohpunkte:
        # a) Tauschhandel Wärme (Robinsons Holz für das gemeinsame Lagerfeuer) gegen Freitags Alternativgüter, und 
        # b) Kooperation bei der Holzentnahme (abhängig vom jeweiligen delta_i): Robinson und Freitag können entweder ihre eigene Holzproduktion maximieren. Oder sie können Beschränkungen für die Holzentnahme vereinbaren, um den Wald langfristig zum gegenseitigen Nutzen zu erhalten.
        # Wir beschränken uns auf den Fall, dass alle Verhandlungen scheitern, sobald der Tauschhandel scheitert. D.h. der vereinfachte Autarkie-Nutzen ergibt sich ohne Tauschhandel und ohne Kooperation bei der Holzentnahme.
        # Ohne Kooperation bei der Holzentnahme wird jeder Spieler seinen Nutzen maximieren:
        # a) Ui_holz(Li) = A_warme * Ai_holz * Li**alpha_holz > Ui_alternativ(Li) = A1_alternativ * Li**alpha_alternativ:
        #    Der individuelle Nutzen aus Holzproduktion übersteigt den Alternativnutzen => Jeder maximiert seine Holzproduktion (Li_holz = Li_max) bis der Wald abgeholzt ist (danach Li_holz = 0).
        # b) Ui_holz(Li) = A_warme * Ai_holz * Li**alpha_holz < Ui_alternativ(Li) = A1_alternativ * Li**alpha_alternativ:
        #    Der Alternativnutzen übersteigt den individuellen Nutzen aus Holzproduktion => Jeder maximiert seine Alternativproduktion (Li_holz = 0).
        # Li_holz bei Autarkie ist daher entweder 0 oder L_max.
        L1_holz_autarkie = Li_max if A_warme * A1_holz > A1_alternativ else 0 # A_warme * A1_holz > A1_alternativ: Holzproduktion lohnt sich immer mehr als Alternativproduktion, solange Wald vorhanden ist
        L2_holz_autarkie = Li_max if A_warme * A2_holz > A2_alternativ else 0 # A_warme * A2_holz > A2_alternativ: Holzproduktion lohnt sich immer mehr als Alternativproduktion, solange Wald vorhanden ist
        # Besser wäre es, die Verhandlungen und Drohpunkte separat zu bestimmen:
        # 1. Strikter Autarkie-Nutzen aus "überhaupt keine Kooperation"
        # 2. Damit Maximierung des Spiels "kein Tauschhandel, jeder produziert seine eigene Wärme, aber kooperative Holzentnahme". Die Allokation ergibt den Drohpunkt für Punkt 3.
        # 3. Maximierung des Spiels "Tauschhandel und kooperative Holznutzung" mit Autarkie-Nutzen aus Punkt 2.

        for delta_i in ((0.1, 0.1), (1.0, 1.0)): # Diskontfaktoren für Robinson und Freitag, z.B. 0.1 für kurzfristige Präferenz, 0.99 für langfristige Präferenz
            delta1 = delta_i[0] # Robinsons Diskontfaktor
            delta2 = delta_i[1] # Freitags Diskontfaktor
            # Optionale Strafe um eine Abholzung des Waldes zu verhindern (externe Regulierung oder kooperative Übereinkunft der Akteure)
            penalties = [ None ] # None: Keine Begrenzung der Abholzung
            def penalty_if_Yi_holz_gt_thresh(x):
                return [100 * max(0, Y1_holz(x[0]) - 0.97), 100 * max(0, Y2_holz(x[1]) - 0.97)] # Beispiel: Strafe für jeden der mehr als 0.97 Holzeinheiten pro Zeit entnimmt
            def penalty_if_Y_holz_gt_thresh(x):
                Y_holz_sum = Y1_holz(x[0]) + Y2_holz(x[1])
                return 100 * max(0, Y_holz_sum - 1.94) * np.array([Y1_holz(x[0]) / Y_holz_sum, Y2_holz(x[1]) / Y_holz_sum]) if Y_holz_sum > 0 else [0, 0] # Beispiel: Anteilige Strafen bei Entnahme von mehr als 1.94 Holzeinheiten pro Zeit
            if A_warme == 4 and delta1 < 1:
                penalties = [ None, penalty_if_Yi_holz_gt_thresh, penalty_if_Y_holz_gt_thresh ]
            for penalty in penalties:
                # Diskontierter Autarkie-Nutzen:
                U1_diskont_autarkie, U2_diskont_autarkie, _, _, _ = U_diskont_fun([L1_holz_autarkie, L2_holz_autarkie, 0], delta1, delta2, penalty, autarkie=True)
                # Nash-Produkt der Nutzenfunktionen, Maximierung führt zu Pareto-effizienter Lösung mit positivem Handelsgewinn für beide
                def nash_product(x): # x[0] = L1_holz, x[1] = L2_holz, x[2] = tau2_alternativ
                    U1_diskont, U2_diskont, _, _, _ = U_diskont_fun(x, delta1, delta2, penalty)
                    return (U1_diskont - U1_diskont_autarkie) * (U2_diskont - U2_diskont_autarkie)
                # Grid-search (Plausibilitätscheck und Startwerte für numerische Lösung)
                def nash_product_grid(x):
                    U1_diskont, U2_diskont, _, _, _ = U_diskont_fun(x, delta1, delta2, penalty)
                    return (U1_diskont - U1_diskont_autarkie) * (U2_diskont - U2_diskont_autarkie) if (U1_diskont > U1_diskont_autarkie) and (U2_diskont > U2_diskont_autarkie) else 0
                sol_grid_x, sol_grid_U = rcw_02_grid_search_iter(U_fun = nash_product_grid, grid_x_start = [0, 0, 0], grid_x_stop = [Li_max, Li_max, 1], grid_x_steps = [0.1, 0.1, 0.1], step_type=GridStepType.DELTA_X, num_iter = 3, verbose=0, constraints_fun = None)
                print(f"  Tauschhandel mit begrenztem Wald, delta1={delta1:.2f}, delta2={delta2:.2f}, L1_holz_autarkie={L1_holz_autarkie:.2f}, L2_holz_autarkie={L2_holz_autarkie:.2f}, U1_diskont_autarkie={U1_diskont_autarkie:.2f}, U2_diskont_autarkie={U2_diskont_autarkie:.2f}, grid-search: x={fmt(sol_grid_x,'{:.2f}')}, Nashprod={sol_grid_U:.2f}")
                # Numerische Lösung, siehe https://docs.scipy.org/doc/scipy/tutorial/optimize.html für Details
                sol_equilibrium = scipy.optimize.minimize(fun = lambda x: -nash_product(x), x0 = sol_grid_x, bounds = ((0, Li_max), (0, Li_max), (0, 1)), 
                    method = "SLSQP", options = {"maxiter": 1000, "ftol": 1e-6}, # Optionen für SLSQP, scipy.optimize.show_options(solver="minimize", method="SLSQP"): ftol: Precision goal for the value of f in the stopping criterion, eps: Step size used for numerical approximation of the Jacobian, maxiter: Maximum number of iterations
                    constraints = [{"type": "ineq", "fun": lambda x: U_diskont_fun(x, delta1, delta2, penalty)[0] - U1_diskont_autarkie}, {"type": "ineq", "fun": lambda x: U_diskont_fun(x, delta1, delta2, penalty)[1] - U2_diskont_autarkie}]) # Randbedingungen: U1_diskont(x) >= U1_diskont_autarkie, U2_diskont(x) >= U2_diskont_autarkie
                if sol_equilibrium.success:
                    sol_x = sol_equilibrium.x
                    U1_diskont, U2_diskont, values, K_wald_tmax, t_exploited = U_diskont_fun(sol_x, delta1, delta2, penalty)
                    sum_U_undiskont = sum(values["U1"]) + sum(values["U2"])
                    sum_Y_undiskont = sum(values["Y1_holz"]) + sum(values["Y2_holz"])
                    penalty_str = f"" if penalty is None else f", {penalty.__name__}"
                    print(f"  Tauschhandel mit begrenztem Wald, delta1={delta1:.2f}, delta2={delta2:.2f}{penalty_str}: L1_holz={sol_x[0]:.2f}, L2_holz={sol_x[1]:.2f}, tau2_alternativ={sol_x[2]:.2f}, t(K_wald=0){t_exploited}, K_wald(t={t_max})={K_wald_tmax:.2f}, U1_diskont={U1_diskont:.2f}, U2_diskont={U2_diskont:.2f}, Nashprod={nash_product(sol_x):.2f}, Sum(Y_holz(t))={sum_Y_undiskont:.2f}, Sum(U1(t)+U2(t))={sum_U_undiskont:.2f}")
                    if A_warme == 4 and penalty is None:
                        if plt_fig is None:
                            plt_fig, _ = plt.subplots(1, 2, figsize=(15,5))
                            plt_idx = 1
                            plt_fig.suptitle(f"Robinson-Crusoe-Holzwirtschaft, Tauschhandel, begrenzter Wald: A_warme={A_warme}, K_wald_max={K_wald_max}, g_wald={g_wald}")
                        plt.subplot(1, 2, plt_idx)
                        plt.plot(values["t"], values["L1_holz"], label=f"L1_holz")
                        plt.plot(values["t"], values["L2_holz"], label=f"L2_holz")
                        plt.plot(values["t"], values["U1"], label=f"U1")
                        plt.plot(values["t"], values["U2"], label=f"U2")
                        plt.plot(values["t"], [max(0,k) for k in values["K"]], label=f"K_wald")
                        plt.xlabel("Zeit t")
                        plt.ylabel("U1, U2, K_wald")
                        plt.legend()
                        plt.title(f"δ1={delta1}, δ2={delta2}, L1_holz={sol_x[0]:.2f}, L2_holz={sol_x[1]:.2f}, t(K_wald=0){t_exploited}")
                        plt_idx += 1
                else:
                    print(f"  Tauschhandel mit begrenztem Wald: scipy.optimize.minimize failed (\"{sol_equilibrium.message}\")")

if __name__ == "__main__":

    # Nash-Gleichgewichte am Beispiel Holzproduktion und gemeinschaftlichem Holzkonsum (Lagerfeuer)
    rcw_02_wood_production()
    print(f"\nmini_wm_robinson_forrest finished.")
    plt.show()
