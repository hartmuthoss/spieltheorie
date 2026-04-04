"""
Das klassische Gefangenendilemma unendlich oft wiederholt.
Erläuterungen siehe WiederholtesGefangenendilemma.md
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import numpy as np
import pathlib
from sympy import *
import spb # pip install sympy_plot_backends

# Rekursiver Aufruf von solveset über alle AND- und OR-Relationen eines gegebenen Terms. Liefert eine Menge von Intervallen über q, die den gegebenen Term erfüllen.
# https://docs.sympy.org/latest/modules/solvers/solveset.html: solveset returns a Set representing all of the solutions of a univariate equation.
def solveset_recursive(q_condition, q, q_interval):
    if q_condition == S.true: # Bedingung immer erfüllbar
        return q_interval
    elif q_condition == S.false: # Bedingung niemals erfüllbar
        return S.EmptySet
    elif isinstance(q_condition, And): # AND: intersect über alle rekursiven Argumente
        q_set = q_interval
        for arg in q_condition.args:
            q_subset = solveset_recursive(arg, q, q_interval)
            q_set = q_set.intersect(q_subset)
        return q_set
    elif isinstance(q_condition, Or): # OR: union über alle rekursiven Argumente
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

# Allgemeine Lösung für q: Sucht ein oder mehrere Intervalle von q, die eine gegebene Bedingung für q erfüllen 
def search_for_q_intervals(q_condition, q, q_interval):
    if q_condition == S.true: # Bedingung immer erfüllbar
        return [ q_interval ]
    elif q_condition == S.false: # Bedingung niemals erfüllbar
        return []
    else:
        assert(isinstance(q_condition, And) or isinstance(q_condition, Or) or q_condition == S.true or q_condition.contains(q))
        # Allgemeine Lösung einer gegebenen Bedingung für q durch solveset:
        # https://docs.sympy.org/latest/modules/solvers/solveset.html: solveset returns a Set representing all of the solutions of a univariate equation.
        q_sol = solveset_recursive(q_condition, q, q_interval)
        # Fallback, falls solveset fehlschlägt: Lösung durch solve. solve() garantiert aber nicht, dass eine oder alle möglichen Lösungen gefunden werden, siehe https://docs.sympy.org/latest/modules/solvers/solveset.html: 
        # There are cases where solve returns an empty list. This might mean that there are no solutions or no solution could be found given its currently supported features. 
        # In other cases, there is no way (given the output interface of solve) to communicate whether or not:
        # * all possible solutions to the system were found,
        # * or that there are provably no solutions,
        # * or that the real set of solutions are finite or infinite,
        # * etc.
        # * solve may return a few solutions when more solutions (potentially infinitely many) also exist.
        # q_sol = solve(list(q_condition.args), q)
        if q_sol == S.true: # Bedingung für beliebige q immer erfüllbar
            return [ True, (q_interval.inf + q_interval.sup) / 2, q_interval ]
        elif q_sol == S.false or q_sol == S.EmptySet: # Bedingung niemals für irgendein q erfüllbar
            return []
        elif isinstance(q_sol, Interval): # Bedingung durch ein einzelnes Intervall über q erfüllbar
            return [ q_sol ]
        elif isinstance(q_sol, Union): # Bedingung durch mehrere Intervalle über q erfüllbar
            return [ q_interval for q_interval in q_sol.args if q_interval.sup - q_interval.inf > 1.0e-6 ]
        elif isinstance(q_sol, FiniteSet): # Bedingung durch ein oder mehrere Intervalle über q erfüllbar
            return [ Interval(q_value, q_value) for q_value in q_sol.args if q_interval.contains(q_value) ]
        else:
            assert(False)
            return [] 
    return []

# Berechnung der reellen Schnittpunkte von Kurven mit
# functions[n][0] = Symbol x der n-ten Funktion, 
# functions[n][1] = Symbolischer Ausdruck f(x) der n-ten Funktion, 
# functions[n][2] = Textlabel der n-ten Funktion
def curve_intersections(functions, x, x_interval=Interval(0, 1)):
    all_intersections = []
    for i in range(len(functions)):
        for j in range(i + 1, len(functions)):
            fi = functions[i]
            fj = functions[j]
            # Reelle Schnittpunkte der Kurven fi und fj
            yi = fi[1].subs(fi[0], x)
            yj = fj[1].subs(fj[0], x)
            sol = solveset(Eq(yi, yj), x, domain=S.Reals)
            for sol_x in sol.args:
                if x_interval.contains(sol_x):
                    yif = yi.subs(x, sol_x).evalf()
                    yjf = yj.subs(x, sol_x).evalf()
                    assert(math.isclose(yif, yjf, rel_tol=1e-6, abs_tol=1e-6))
                    all_intersections.append((yi, yj, sol_x,))
                    print(f"  Schnittpunkt {yi} = {yj} für {x} in {x_interval}: {x} = {sol_x} = {sol_x.evalf():.4f}, f({x}) = {yif:.4f}")
    return all_intersections

# Plot von Funktionen und labeln mit 
# functions[n][0] = Symbol x der n-ten Funktion, 
# functions[n][1] = Symbolischer Ausdruck f(x) der n-ten Funktion, 
# functions[n][2] = Textlabel der n-ten Funktion
def plot_example(x_values, functions, title="", x_label="", y_label="", png_file="", figsize=(9, 9)):
    fig = plt.figure(figsize=figsize)
    for f in functions:
        y_values = [ f[1].subs(f[0], x_value) for x_value in x_values ]
        plt.plot(x_values, y_values, label=f[2])
    if x_label != "":
        plt.xlabel(x_label)
    if y_label != "":
        plt.ylabel(y_label)
    if title != "":
        plt.title(title)
    plt.legend()
    if png_file != "":
        pathlib.Path(png_file).unlink(missing_ok=True)
        plt.savefig(png_file)
        title = title.replace("\n", " ")
        print(f"  Plot gespeichert in {png_file}")

# Beispiel 1: Wiederholtes Gefangenendilemma, jeder Spieler wiederholt die Antwort des anderen aus der vorangegangenen Runde.
def example_01_repeat_answer():
    title = "Unendlich oft wiederholtes Gefangenendilemma:\nSpiegeln der gegenseitigen Antworten"
    x = symbols("x", real = True)
    x_values = np.arange(0, 0.9, 1e-3)
    functions = [
        ( x, 3 / (1 - x), "U1 mit Start (Leugnen, Leugnen)\nU2 mit Start (Leugnen, Leugnen)" ),
        ( x, 2 / (1 - x), "U1 mit Start (Gestehen, Gestehen)\nU2 mit Start (Gestehen, Gestehen)" ),
        ( x, 5 / (1 - x*x) - 4, "U1 mit Start (Leugnen, Gestehen)\nU2 mit Start (Gestehen, Leugnen)" ),
        ( x, 5 / (1 - x*x) - 1, "U2 mit Start (Leugnen, Gestehen)\nU1 mit Start (Gestehen, Leugnen)" ),
        ]
    print(f"\n{title}:")
    curve_intersections(functions=functions, x=x, x_interval=Interval(0, 1))
    plot_example(x_values=x_values, functions=functions, title=title, x_label="Diskontfaktor δ", y_label="Nutzen U", png_file="prisoners_dilemma_repeated_01.png")

# Beispiel 2: Wiederholtes Gefangenendilemma, ein Gefangener spielt Tit for Tat, der andere spielt entweder ebenfalls Tit for Tat, 
# oder er ist ein Kooperateur (der stets leugnet), oder er ist ein Verräter (der stets gesteht).
def example_02_tit_for_tat():
    title = "Unendlich oft wiederholtes Gefangenendilemma:\nTit for Tat gegen Kooperateur, Tit for Tat oder Verräter"
    x = symbols("x", real = True)
    x_values = np.arange(0, 0.9, 1e-3)
    functions = [
        ( x, 3 / (1 - x), "U1, U2 bei Tit for Tat gegen Kooperateur\nU1, U2 bei Tit for Tat gegen Tit for Tat" ),
        ( x, 2 / (1 - x) - 1, "U1 bei Tit for Tat gegen Verräter" ),
        ( x, 2 / (1 - x) + 2, "U2 bei Tit for Tat gegen Verräter" ),
        ]
    print(f"\n{title}:")
    curve_intersections(functions, x=x, x_interval=Interval(0, 1))
    plot_example(x_values=x_values, functions=functions, title=title, x_label="Diskontfaktor δ", y_label="Nutzen U", png_file="prisoners_dilemma_repeated_02.png")

# Beispiel 2: Wiederholtes Gefangenendilemma mit Tit for Tat, Check von teilspielperfekten Gleichgewichten (subgame perfect equilibrium, SPE) mit One-Shot Deviation Principle (OSDP)
def example_02_tit_for_tat_OSDP():
    # SPE-Bedingung für Spieler 1: Spieler 1 hat keinen Nutzen von einmaliger Abweichung, wenn U_1(1) ≥ U_1(3a) ≥ U_1(3b)
    # Im Beispiel:
    # U_1(1) ≥ U_1(3a) ⇔ 3 / (1 - δ) ≥ (4 + δ) / (1 - δ^2)   ⇔ 1/2 ≤ δ < 1
    # U_1(1) ≥ U_1(3b) ⇔ 3 / (1 - δ) ≥ 1 - 2 δ + 3 / (1 - δ) ⇔ 1/2 ≤ δ < 1
    # U_1(3a) ≥ U_1(3b) ⇔ (4 + δ) / (1 - δ^2) ≥ 1 - 2 δ + 3 / (1 - δ) ⇔ 0 ≤ δ ≤ 1/2
    # SPE-Bedingung für Spieler 2 entsprechend.
    d = symbols("d", real = True)
    SPE_conditions = [ [ Ge(3 / (1 - d), (4 + d) / (1 - d*d)), Ge(d, 0), Lt(d, 1) ],              # U_1(1)  ≥ U_1(3a) ∧ 0 <= d < 1
                       [ Ge(3 / (1 - d), 1 - 2 * d + 3 / (1 - d)), Ge(d, 0), Lt(d, 1) ],          # U_1(1)  ≥ U_1(3b) ∧ 0 <= d < 1
                       [ Ge((4 + d) / (1 - d*d), 1 - 2 * d + 3 / (1 - d)), Ge(d, 0), Lt(d, 1) ] ] # U_1(3a) ≥ U_1(3a) ∧ 0 <= d < 1
    # Liste über Intervalle d, die die SPE_conditions mit 0 <= d < 1 erfüllen
    print(f"\nUnendlich oft wiederholtes Gefangenendilemma mit Tit for Tat: Check SPEs mit One-Shot Deviation Principle (OSDP)")
    for n, SPE_condition in enumerate(SPE_conditions):
        SPE_sol = reduce_inequalities(SPE_condition, d)
        SPE_sol_intervals = search_for_q_intervals(SPE_sol, d, Interval(0, 1, False, True))
        print(f"  {n+1}. SPE-Bedingung = {SPE_condition}, 0<=d<1")
        print(f"  {n+1}. SPE-Bedingung <=> {SPE_sol}")
        print(f"  {n+1}. SPE-Bedingung <=> {SPE_sol_intervals}")
    # Lösung für alle Bedingungen
    SPE_condition = sum(SPE_conditions, []) # Flat list
    SPE_sol = reduce_inequalities(SPE_condition)
    SPE_sol_intervals = search_for_q_intervals(SPE_sol, d, Interval(0, 1, False, True))
    print(f"  Alle SPE-Bedingungen = {SPE_condition}, 0<=d<1")
    print(f"  Alle SPE-Bedingungen <=> {SPE_sol}")
    print(f"  Alle SPE-Bedingungen <=> {SPE_sol_intervals}")

# Beispiel 3: Wiederholtes Gefangenendilemma, ein Gefangener spielt Grim Trigger, der andere spielt entweder ebenfalls Grim Trigger
# oder Tit for Tat, oder er ist ein Kooperateur (der stets leugnet), oder er ist ein Verräter (der stets gesteht).
def example_03_grim_trigger():
    title = "Unendlich oft wiederholtes Gefangenendilemma:\nGrim Trigger gegen Kooperateur, Tit for Tat, Grim Trigger oder Verräter"
    x = symbols("x", real = True)
    x_values = np.arange(0, 0.9, 1e-3)
    functions = [
        ( x, 3 / (1 - x), "U1, U2 bei Grim Trigger gegen Kooperateur,\ngegen Tit for Tat oder gegen Grim Trigger" ),
        ( x, 2 / (1 - x) - 1, "U1 bei Grim Trigger gegen Verräter" ),
        ( x, 2 / (1 - x) + 2, "U2 bei Grim Trigger gegen Verräter" ),
        ]
    print(f"\n{title}:")
    curve_intersections(functions, x=x, x_interval=Interval(0, 1))
    plot_example(x_values=x_values, functions=functions, title=title, x_label="Diskontfaktor δ", y_label="Nutzen U", png_file="prisoners_dilemma_repeated_03.png")

# Beispiel 3: Wiederholtes Gefangenendilemma mit Grim Trigger, Check von teilspielperfekten Gleichgewichten (subgame perfect equilibrium, SPE) mit One-Shot Deviation Principle (OSDP)
def example_03_grim_trigger_OSDP():
    # Spieler 1 hat keinen Nutzen von einmaliger Abweichung, wenn $U_1(1) ≥ U_1(3) ∧ U_1(1) ≥ U_1(4)$  
    # Im Beispiel: $3 / (1 - δ) ≥ 2 + 2 / (1 - δ) ∧ 3 / (1 - δ) ≥ 2 / (1 - δ) ⇔ δ ≥ 1/2$  
    # Spieler 2 hat keinen Nutzen von einmaliger Abweichung, wenn $U_2(1) ≥ U_2(2) ∧ U_2(1) ≥ U_2(4)$  
    # Im Beispiel: $3 / (1 - δ) ≥ 2 + 2 / (1 - δ) ∧ 3 / (1 - δ) ≥ 2 / (1 - δ) ⇔ δ ≥ 1/2$  
    # Allgemein: $R / (1 - δ) ≥ T - P + P / (1 - δ) ∧ R / (1 - δ) ≥ P / (1 - δ) ⇔ δ ≥ (T - R)  / (T - P) ∧ R ≥ P$  
    d = symbols("d", real = True)
    SPE_conditions = [ [ Ge(3 / (1 - d), 2 + 2 / (1 - d)), Ge(3 / (1 - d), 2 / (1 - d)) ], ]
    # Liste über Intervalle d, die die SPE_conditions mit 0 <= d < 1 erfüllen
    print(f"\nUnendlich oft wiederholtes Gefangenendilemma mit Grim Trigger: Check SPEs mit One-Shot Deviation Principle (OSDP)")
    for n, SPE_condition in enumerate(SPE_conditions):
        SPE_sol = reduce_inequalities(SPE_condition, d)
        SPE_sol_intervals = search_for_q_intervals(SPE_sol, d, Interval(0, 1, False, True))
        print(f"  {n+1}. SPE-Bedingung <=> {SPE_condition}, 0<=d<1")
        print(f"  {n+1}. SPE-Bedingung <=> {SPE_sol}")
        print(f"  {n+1}. SPE-Bedingung <=> {SPE_sol_intervals}")

# Suche nach Lösungen von SPE-Bedingungen mittels reduce_inequalities, search_for_q_intervals und evalf.
def SPE_solutions(title, SPE_condition, d):
    SPE_sol = reduce_inequalities(SPE_condition, d)
    SPE_sol_intervals = search_for_q_intervals(SPE_sol, d, Interval(0, 1, False, True))
    SPE_sol_intervals_evalf = [ Interval(interval.inf.evalf(), interval.sup.evalf(), interval.left_open, interval.right_open) if not interval.is_FiniteSet else interval for interval in SPE_sol_intervals ]
    print(f"{title} = {SPE_condition}, 0<=d<1")
    print(f"{title} <=> {SPE_sol}")
    print(f"{title} <=> {SPE_sol_intervals}")
    print(f"{title} <=> {SPE_sol_intervals_evalf}")
    return SPE_sol_intervals_evalf

# Ruft evalf() rekursiv über alle Teilausdrücke eines Terms x auf
def evalf_recursive(expr, d):
    y = expr
    if isinstance(expr, And):
        y = And(True)
        for x in expr.args:
            y = And(evalf_recursive(x, d), y)
    elif isinstance(expr, Or):
        y = Or(False)
        for x in expr.args:
            y = Or(evalf_recursive(x, d), y)
    elif expr.is_Boolean:
        y = expr
    else:
        y = expr.evalf()
    return y

# Beispiel 4: Wiederholtes Gefangenendilemma mit Tit for Tat mit N-facher Strafe und Check von Abweichbedingungen mit One-Shot Deviation Principle (OSDP)
def example_04_N_tit_for_tat_OSDP(png_file="prisoners_dilemma_repeated_04.png", figsize=(8, 8)):
    labels = [ "U1(1) ≥ U1(3a)", "U1(1) ≥ U1(3b) ∧ U1(3a) ≥ U1(3b)", "U1(1) ≥ U1(3c) ∧ U1(3c) ≥ U1(3a)", "U1(1) ≥ U1(3c) ∧ U1(3b) ≥ U1(3c)", "MIT-1" ] # Label der Abweichbedingungen
    d_plot_lower_bounds = { label: [] for label in labels }
    d_plot_upper_bounds = { label: [] for label in labels }
    N_plot_values = { label: [] for label in labels }
    N_values = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20 ] # [ 1, 2, 3, 5, 10 ]
    for N in N_values:
        # Zustandsfolgen und Nutzen in Abhängigkeit von δ
        d = symbols("d", real = True)
        U1_1 = 3 / (1 - d) # Zustandsfolge 1, Spieler 1: NTfT, Spieler 2: NTfT, (L, L) → (L, L), U1 = U2 = R / (1 - δ) = 3 / (1 - δ), beiderseitige Kooperation
        U1_3a = (4 + d) / (1 - d*d) # Zustandsfolge 3a, Spieler 1: NTfTAbw, Spieler 2: NTfT, ohne Rückkehr zur Kooperation, N=1, (G, L) → (L, G) → (G, L), U1 = (T + S*δ) / (1 - δ^2) = (4 + δ) / (1 - δ^2), wie klassisches Tit for Tat
        U1_3b = 4 + 2 * (d - d**N) / (1 - d) + d**N / (1 - d*d) + 4 * d**(N+1) / (1 - d*d) # Zustandsfolge 3b, Spieler 1: NTfTAbw, Spieler 2: NTfT, mit Rückkehr zu klassischem Tit for Tat nach Strafphase, N>0, (G, L) → { (N-1)-mal (G, G) } → (L, G) → (G, L) → (L, G)
        U1_3c = 4 + 2 * (d - d**N) / (1 - d) + d**N + 3 * d**(N+1) / (1 - d)               # Zustandsfolge 3c, Spieler 1: NTfTAbw, Spieler 2: NTfT, mit Rückkehr zur Kooperation,  N>0, (G, L) → { (N-1)-mal (G, G) } → (L, G) → (L, L) → (L, L)
        U1_3d = 4 + 2 * (d - d**(N+1)) / (1 - d) + 3 * d**(N+1) / (1 - d)                  # MIT Strategie 1,  Spieler 1: NTfTAbw, Spieler 2: NTfT, mit Rückkehr zur Kooperation,  N>0, (G, L) → { N-mal (G, G) } → (L, L) → (L, L)
        # Abweichbedingungen (nicht unbedingt SPE)
        SPE_conditions = [ [ Ge(U1_1,  U1_3a), Ge(d, 0), Lt(d, 1) ],                    # U_1(1) ≥ U_1(3a) ∧ 0 <= d < 1
                           [ Ge(U1_1,  U1_3b), Ge(U1_3a, U1_3b), Ge(d, 0), Lt(d, 1) ],  # U_1(1) ≥ U_1(3b) ∧ U_1(3a) ≥ U_1(3b) ∧ 0 <= d < 1
                           [ Ge(U1_1,  U1_3c), Ge(U1_3c, U1_3a), Ge(d, 0), Lt(d, 1) ],  # U_1(1) ≥ U_1(3c) ∧ U_1(3a) ≥ U_1(3c) ∧ 0 <= d < 1
                           [ Ge(U1_1,  U1_3c), Ge(U1_3b, U1_3c), Ge(d, 0), Lt(d, 1) ] ] # U_1(1) ≥ U_1(3c) ∧ U_1(3b) ≥ U_1(3c) ∧ 0 <= d < 1
        # Liste über Intervalle d, die die SPE_conditions mit 0 <= d < 1 erfüllen
        print(f"\nUnendlich oft wiederholtes Gefangenendilemma mit Tit for Tat mit N-facher Strafe: Check Abweichbedingungen")
        for n, SPE_condition in enumerate(SPE_conditions):
            print(f"  N={N}, {n+1}. SPE-Bedingung: {labels[n]}")
            SPE_intervals = SPE_solutions(f"  N={N}, {n+1}. SPE-Bedingung", SPE_condition, d)
            label = labels[n]
            N_plot_values[label].append(N)
            d_plot_lower_bounds[label].append(min([interval.inf for interval in SPE_intervals]))
            d_plot_upper_bounds[label].append(max([interval.sup for interval in SPE_intervals]))
        # Lösung für alle Abweichbedingungen
        # SPE_condition = sum(SPE_conditions, []) # Flat list aller SPE-Bedingungen
        # SPE_solutions(f"  N={N}, Alle SPE-Bedingungen", SPE_condition, d)
        # Vergleich mit MIT Strategien 1 (similar to grim trigger) und 2 (similar to tit-for-tat) in https://ocw.mit.edu/courses/17-810-game-theory-spring-2021/mit17_810s21_lec5.pdf: 
        # MIT Strategie 1 (Similar to grim trigger): 
        #   "Cooperate until your opponent defects. If your opponent defects, do not cooperate for the next k periods but then return to cooperation; 
        #   if you defect, do not cooperate for the next k periods but then return to cooperation. Once you have returned to cooperation, cooperate until a defection occurs.
        #   Consequently, sustaining cooperation requires that: a / (1 - δ) ≥ c + b * (δ - δ^{N+1}) / (1 - δ) + a * (δ^{N+1}) / (1 - δ)
        #   We cannot generate a closed form for the critical value of δ, but we can rewrite this expression as: δ ≥ (c−a) / (a−d) + δ^{N+1} \cdot (a-b) / (c-b)"
        #   Im Beispiel mit a=R=3, b=P=2, c=T=4, d=S=1: δ ≥ 1/2 + δ^{N+1} / 2
        # MIT Strategie 2 (Similar to tit-for-tat): 
        #   "Cooperate until your opponent defects. If your opponent defects, do not cooperate for k periods. If she cooperates in any of the k periods, return to cooperation, 
        #   ending the punishment phase. If she fails to cooperate in any period of the punishment phase, then the punishment phase starts over i.e. don’t cooperate for k more periods. 
        #   If your own failure to cooperate caused the punishment phase then cooperate during the punishment phase.
        #   A defection from the cooperation phase generates a payoff consisting of a one period benefit c, a punishment payoff of d for k periods, and a return to cooperative payoffs a 
        #   at the end of the punishment. 
        #   Summing these up generates a payoff c + d * (δ - δ^{N+1}) / (1 - δ) + a * (δ^{N+1}) / (1 - δ)
        #   Thus, increasing the length of the punishment phase decreases the incentive to defect from the cooperative phase."
        #   => Bedingung: a / (1 - δ) ≥ c + d * (δ - δ^{N+1}) / (1 - δ) + a * (δ^{N+1}) / (1 - δ)
        #   Im Beispiel mit a=R=3, b=P=2, c=T=4, d=S=1: 3 / (1 - δ) ≥ 4 + (δ - δ^{N+1}) / (1 - δ) + 3 * (δ^{N+1}) / (1 - δ)
        # MIT_strategy_1_condition = [ Ge(d, 1/2 + 1/2 * d**(N+1)), Ge(d, 0), Lt(d, 1) ]
        MIT_strategy_1_condition = [ Ge(3 / (1 - d), 4 + 2 * (d - d**(N+1)) / (1 - d) + 3 * (d**(N+1)) / (1 - d)), Ge(d, 0), Lt(d, 1) ]
        MIT_strategy_1_intervals = SPE_solutions(f"  N={N}, MIT-Strategie 1, similar to grim trigger", MIT_strategy_1_condition, d)
        label = "MIT-1"
        if len(MIT_strategy_1_intervals) > 0:
            N_plot_values[label].append(N)
            d_plot_lower_bounds[label].append(min([interval.inf for interval in MIT_strategy_1_intervals]))
            d_plot_upper_bounds[label].append(max([interval.sup for interval in MIT_strategy_1_intervals]))
        if N <= 10:
            MIT_strategy_2_condition = [ Ge(3 / (1 - d), 4 + (d - d**(N+1)) / (1 - d) + 3 * (d**(N+1)) / (1 - d)), Ge(d, 0), Lt(d, 1) ]
            MIT_strategy_2_intervals = SPE_solutions(f"  N={N}, MIT-Strategie 2, similar to tit-for-tat", MIT_strategy_2_condition, d)
    # Plot der obere Grenzen der δ-Intervalle über N für unterschiedliche Abweichbedingungen
    fig = plt.figure(figsize=figsize)
    plot_x, plot_y, plot_label = [], [], []
    for n, label in enumerate(labels):
        if d_plot_lower_bounds[label] in plot_y:
            i = plot_y.index(d_plot_lower_bounds[label])
            plot_label[i] += f"\nδ1, {label}"
        else:
            plot_x.append(N_plot_values[label])
            plot_y.append(d_plot_lower_bounds[label])
            plot_label.append(f"δ1, {label}")
        if d_plot_upper_bounds[label] in plot_y:
            i = plot_y.index(d_plot_upper_bounds[label])
            plot_label[i] += f"\nδ2, {label}"
        else:
            plot_x.append(N_plot_values[label])
            plot_y.append(d_plot_upper_bounds[label])
            plot_label.append(f"δ2, {label}")
    for n in range(len(plot_x)):
        plt.plot(plot_x[n], plot_y[n], label=plot_label[n])
    plt.xticks(range(N_values[0],N_values[-1]+1))
    plt.xlabel(f"N")
    plt.ylabel(f"δ1, δ2")
    plt.title(f"Unendlich oft wiederholtes Gefangenendilemma:\nδ-Intervalle bei Tit for Tat mit N-facher Strafe")
    plt.legend()
    if png_file != "":
        pathlib.Path(png_file).unlink(missing_ok=True)
        plt.savefig(png_file)
        print(f"  Plot gespeichert in {png_file}")

# Beispiel 5a: Wiederholtes Gefangenendilemma, beide Gefangene spielen Tit for Tat mit Antesten. Jeder Spieler ist im Ausgangspunkt kooperativ, 
# testet den Gegenspieler aber mit einer Wahrscheinlichkeit q durch Verrat an. 3D- und 2D-plots der Nutzenfunktionen für q1 = q2 = q.
def example_05a_tit_for_tat_modified(title = "Unendlich oft wiederholtes Gefangenendilemma: Tit for Tat mit Antesten", png_file="prisoners_dilemma_repeated_05a.png", figsize=(18, 9)):
    fig = plt.figure(figsize=figsize)
    # Nutzenfunktion U(q,δ)
    U = lambda q, d : q * (1-q) * (6 + 5 * d + 5 * d*d) / (1 - d*d*d) + q*q * (1 + 2 / (1 - d)) + (1-q) * (1-q) * 3 / (1 - d)
    # 3D-plot der Nutzenfunktion U über q und δ
    ax = fig.add_subplot(121, projection='3d')
    q = np.arange(0, 1.0, 1e-3) # Wahrscheinlichkeit q
    d = np.arange(0, 0.9, 1e-3) # Diskontfaktor δ
    q, d = np.meshgrid(q, d)    # q,d-grid
    surf = ax.plot_surface(q, d, U(q, d), cmap=cm.coolwarm) # 3D-plot der Nutzenfunktion U(q,δ)
    fig.colorbar(surf, shrink=0.5)
    ax.set_xlabel("Wahrscheinlichkeit q = P(Verrat)")
    ax.set_ylabel("Diskontfaktor δ")
    ax.set_zlabel("Nutzen U(q,δ)")
    plt.tight_layout()
    # 2D-plot der Nutzenfunktion U(δ) für q = 0, 0.5, 1
    ax = fig.add_subplot(122)
    bbox = ax.get_position()
    ax.set_position([bbox.bounds[0], 0.1 + bbox.bounds[1], 0.7 * bbox.bounds[2], 0.7 * bbox.bounds[3]])
    d = np.arange(0, 0.9, 1e-3) # Diskontfaktor δ
    for q in [ 0, 0.5, 1 ]:     # Wahrscheinlichkeit q
        ax.plot(d, U(q, d), label=f"q={q}") # 2D-plot der Nutzenfunktion U(δ) bei gegebenem q
    plt.xlabel("Diskontfaktor δ")
    plt.ylabel("Nutzen U")
    plt.legend()
    plt.suptitle(title)
    if png_file != "":
        pathlib.Path(png_file).unlink(missing_ok=True)
        plt.savefig(png_file)
        title = title.replace("\n", " ")
        print(f"{title}:\n  Plot gespeichert in {png_file}")

# Beispiel 5b: Wiederholtes Gefangenendilemma, beide Gefangene spielen Tit for Tat mit Antesten. Jeder Spieler ist im Ausgangspunkt kooperativ, 
# testet den Gegenspieler aber mit einer Wahrscheinlichkeit q durch Verrat an (Spieler 1 mit q1, Spieler 2 mit q2). Jeder Spieler bestimmt qi so,
# dass sein Nutzen maximal wird.
def example_05b_tit_for_tat_modified(title = ["Unendlich oft wiederholtes Gefangenendilemma:", "Tit for Tat mit Antesten"], png_file="prisoners_dilemma_repeated_05b.png"):
    print(f"{title[0]} {title[1]}:")
    # Nutzenfunktionen U1(q1, q2, δ) und U2(q1, q2, δ)
    q1, q2, d = symbols("q1 q2 d", real = True)
    U1 = q1 * (1-q2) * (3 + 4 * d + d*d) / (1 - d*d*d) + (1-q1) * q2 * (3 + d + 4 * d*d) / (1 - d*d*d) + q1 * q2 * (1 + 2 / (1 - d)) + (1-q1) * (1-q2) * 3 / (1 - d)
    U2 = q1 * (1-q2) * (3 + d + 4 * d*d) / (1 - d*d*d) + (1-q1) * q2 * (3 + 4 * d + d*d) / (1 - d*d*d) + q1 * q2 * (1 + 2 / (1 - d)) + (1-q1) * (1-q2) * 3 / (1 - d)
    print(f"  U1 = {U1}")
    print(f"  U2 = {U2}")
    # Maxima: ∂U1/∂q1 = 0, ∂U2/∂q2 = 0
    dU1_dq1 = diff(U1, q1)
    dU2_dq2 = diff(U2, q2)
    sol_q1 = solveset(Eq(dU1_dq1, 0), q1, domain=S.Reals)
    sol_q2 = solveset(Eq(dU2_dq2, 0), q2, domain=S.Reals)
    print(f"  dU1/dq1 = {dU1_dq1} hängt{' ' if dU1_dq1.has(q1) else ' nicht '}von q1 ab")
    print(f"  dU2/dq2 = {dU2_dq2} hängt{' ' if dU2_dq2.has(q2) else ' nicht '}von q2 ab")
    # U1, U2 für die Randfälle (q_1 = q_2 = 0), (q_1 = 1, q_2 = 0), (q_1 = 0, q_2 = 1), (q_1 = q_2 = 1)
    U1_0_0 = simplify(U1.subs({q1: 0, q2: 0}))
    U2_0_0 = simplify(U2.subs({q1: 0, q2: 0}))
    U1_0_1 = simplify(U1.subs({q1: 0, q2: 1}))
    U2_0_1 = simplify(U2.subs({q1: 0, q2: 1}))
    U1_1_0 = simplify(U1.subs({q1: 1, q2: 0}))
    U2_1_0 = simplify(U2.subs({q1: 1, q2: 0}))
    U1_1_1 = simplify(U1.subs({q1: 1, q2: 1}))
    U2_1_1 = simplify(U2.subs({q1: 1, q2: 1}))
    print(f"  U1(q_1 = 0, q_2 = 0) = {U1_0_0}")
    print(f"  U2(q_1 = 0, q_2 = 0) = {U2_0_0}")
    print(f"  U1(q_1 = 0, q_2 = 1) = {U1_0_1}")
    print(f"  U2(q_1 = 0, q_2 = 1) = {U2_0_1}")
    print(f"  U1(q_1 = 1, q_2 = 0) = {U1_1_0}")
    print(f"  U2(q_1 = 1, q_2 = 0) = {U2_1_0}")
    print(f"  U1(q_1 = 1, q_2 = 1) = {U1_1_1}")
    print(f"  U2(q_1 = 1, q_2 = 1) = {U2_1_1}")
    # Plot U1, U2 für die Randfälle (q_1 = q_2 = 0), (q_1 = 1, q_2 = 0), (q_1 = 0, q_2 = 1), (q_1 = q_2 = 1)
    plots = [None, None]
    for i, d_interval in enumerate([(d,0,0.8), (d,0.8,0.99)]):
        plots[i] = spb.plot((U1_0_0,{"label": "U1(q1=0,q2=0) = U2(q1=0,q2=0)"}), # U1_0_0 = U2_0_0
                 (U1_0_1,{"label": "U1(q1=0,q2=1) = U2(q1=1,q2=0)"}), # U1_0_1 = U2_1_0
                 (U1_1_0,{"label": "U1(q1=1,q2=0) = U2(q1=0,q2=1)"}), # U1_1_0 = U2_0_1
                 (U1_1_1,{"label": "U1(q1=1,q2=1) = U2(q1=1,q2=1)"}), # U1_1_1 = U2_1_1
                 d_interval, title=title[i], xlabel = "Diskontfaktor δ", ylabel = "Nutzen U", legend = True, show = False)
    pltg = spb.plotgrid(plots[0], plots[1], nc = 2, nr = 1, show = False)
    if png_file != "":
        pathlib.Path(png_file).unlink(missing_ok=True)
        pltg.save(png_file)
        print(f"  Plot gespeichert in {png_file}")
    else:
        pltg.show()

if __name__ == "__main__":
    example_01_repeat_answer()
    example_02_tit_for_tat()
    example_02_tit_for_tat_OSDP()
    example_03_grim_trigger()
    example_03_grim_trigger_OSDP()
    example_04_N_tit_for_tat_OSDP()
    example_05a_tit_for_tat_modified()
    example_05b_tit_for_tat_modified()
    plt.show()
    print(f"")
