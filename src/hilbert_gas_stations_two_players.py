"""
Gleichgewichte mit 2 Tankstellen (Hilbert-Tankstellen mit nur 2 Spielern):
Beschreibung und Erlaeuterungen in ../docs/HilbertTankstellen.md
"""
import math
import numpy as np
from enum import IntEnum
from sympy import *
from sympy.assumptions.relation.binrel import AppliedBinaryRelation

# Logging auf stdout und/oder logfile
class HilbertLogger():
    class Verbosity(IntEnum):
        ERROR = 0,
        WARNING = 1,
        INFO = 2,
        DEBUG = 3
    def __init__(self, verbosity = Verbosity.INFO, filename = ""):
        self.verbosity = verbosity
        self.filename = filename
        self.fp = open(self.filename, "w", encoding="utf-8") if self.filename != "" else None
    def print(self, verbosity = Verbosity.INFO, msg: str = ""):
        if verbosity <= self.verbosity:
            print(msg)
            if self.fp is not None:
                self.fp.write(msg + "\n")

# Parameter für Hilbert-Tankstellen (symbolisch oder konkrete Beispielwerte)
class HilbertParameter():
    def __init__(self, symbolic = True, is_established = True, is_rich = True):
        # Konstante Parameter
        self.e_n = symbols("e_n", real=True) if symbolic else    1  # Einkaufspreis, z.B. 1 Euro/liter
        self.b_n = symbols("b_n", real=True) if symbolic else    1  # laufende Betriebskosten, z.B. 1000 Euro/Min
        self.F_n = symbols("F_n", real=True) if symbolic else    0  if is_established else 1 # laufende Abschreibung im Falle von Investitionskosten, z.B. 0 (etablierte Tankstelle) oder 0.5 Euro/Min (Neueröffnung einer Tankstelle)
        self.K_n = symbols("K_n", real=True) if symbolic else    0  if is_rich else 1 # Kosten eines Konkurrenzkampf, z.B. 0 (reiche Tankstelle mit Rücklagen) oder 1 Euro/Min (arme Tankstelle ohne Rücklagen)
        self.s   = symbols("s",   real=True) if symbolic else 0.19  # Umsatzsteuer, z.B. 0.19 (19% Mwst.)
        # Arbeitspunkte: Hebel, Basismenge und Basismengenpreis sind frei konfigurierbar; die verkaufte Gesamtmenge sum(m[n]) reduziert sich linear mit dem durchschnittlichen Verkaufspreis v_mean
        self.m_0 = symbols("m_0", real=True) if symbolic else   10  # Basismenge, z.B. 10 liter
        self.v_0 = symbols("v_0", real=True) if symbolic else    2  # Basismengenpreis, z.B. 2 Euro/liter
        self.h   = symbols("h",   real=True) if symbolic else (1/2)*(self.m_0/self.v_0)  # Hebel in liter/Euro: m = M * (m_0 - h * (v - v_0)), z.B. h = (1/2)*(m_0/v_0) halbiert die verkaufte Menge bei doppeltem Preis
        self.r   = symbols("r",   real=True) if symbolic else  100  # Rate r modelliert wie sensibel die Kunden auf lokale Preisunterschiede reagieren, Verkaufsmenge m[n] proportional zu (m_sum/N) + r * (v_mean - v[n])

# Für alle Argumente in *values: x = 0 if abs(x) < eps else x
def epsilon2zero(*values, eps = 1.0e-6):
    result = tuple(0 if abs(float(x)) < eps else x for x in values)
    return result[0] if len(result) == 1 else result

# Konvertiert ein Predicate (AppliedBinaryRelation) in eine Relation
def convert_atom(atom):
    if isinstance(atom, AppliedBinaryRelation):
        if atom.function.name.casefold() == "ge":
            return Ge(atom.lhs, atom.rhs)
        if atom.function.name.casefold() == "gt":
            return Gt(atom.lhs, atom.rhs)
        if atom.function.name.casefold() == "le":
            return Le(atom.lhs, atom.rhs)
        if atom.function.name.casefold() == "lt":
            return Lt(atom.lhs, atom.rhs)
        if atom.function.name.casefold() == "eq":
            return Eq(atom.lhs, atom.rhs)
    return atom

# Sucht eine Belegung einer gegebenen se_condition durch Einsetzen von Wahrscheinlichkeitswerten aus einem festen grid
def probability_grid_check(se_condition, q, grid_step = 0.01):
    prob_grid = np.array([[0.5+delta, 0.5-delta] for delta in np.arange(0, 0.5+grid_step, grid_step)]).flatten()[1:] # prob_grid = [0.5, 0.51, 0.49, 0.52, 0.48, ..., 1.0, 0.0]
    for grid_q in prob_grid:
        grid_cond = simplify(se_condition.subs(q, grid_q))
        if grid_cond == True:
            return True, grid_q
        se_rels = list(grid_cond.args) if isinstance(grid_cond, And) else [grid_cond]
        if reduce_inequalities(se_rels) == True:
            return True, grid_q
    return False, math.nan, math.nan, math.nan

# Eine Gleichgewichtsbedingung ist erfüllbar, wenn eine gegebene se_condition erfüllbar ist.
# Falls se_condition nicht symbolisch und erfüllbar ist, wird ein konkreter Wert
# für Wahrscheinlichkeit q gesucht, der se_condition löst (sympy solve plus 
# einfacher Test von schrittweisen Wahrscheinlichkeitswerten aus einem festen grid)
def equilibrium_satisfiable(title, se_condition, se_symbolic, q, logger, grid_step = 0.01):
    
    # Randbedingung für Wahrscheinlichkeitswerte: 0 <= q <= 1
    if se_condition.has(q):
        se_condition = se_condition & (q >= 0) & (q <= 1)
    
    # Erfüllbarkeit der Gleichgewichtsbedingung
    if se_condition == True:
        logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht immer erfuellt")
        return True, 0.5
    if se_condition == False:
        logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewichtsbedingung nicht erfuellbar")
        return False, math.nan
    if not se_symbolic:
        se_rels = list(se_condition.args) if isinstance(se_condition, And) else [se_condition]
        if se_condition.has(q):
            se_solved = reduce_inequalities(se_rels, q)
        else:
            se_solved = reduce_inequalities(se_rels) # solve(se_rels)
    else:
        se_solved = satisfiable(se_condition)
    if se_solved == False:
        logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Kein Gleichgewicht, Gleichgewichtsbedingung nicht erfuellbar")
        return False, math.nan
    
    # Numerische Lösung für nicht-symbolische Ungleichungen suchen
    if not se_symbolic:
        # Beispielswerte für Lösungen aus grid
        grid_satisfiable, grid_q = probability_grid_check(se_condition, q, grid_step)
        if not grid_satisfiable == False:
            logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht gdw. {se_solved}, z.B. {q} = {grid_q}")
            return grid_satisfiable, grid_q
        logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht gdw. {se_solved}, aber keine grid-Loesung für {q} gefunden")
        return se_solved, math.nan
    
    logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht gdw. {se_solved}")
    return se_solved, math.nan

# Symmetrisches Nash-Gleichgewicht (Verkaufspreis) für M Kunden und N Tankstellen via "ein Abweichler"-Ansatz.
# Eine Strategie v_sne ist ein symmetrischen Nash-Gleichgewicht (SNE), wenn für jeden Spieler i gilt: v_sne ∈ arg​ max_vi ​Ui​(vi​,v_sne,...,v_sne)
# Oder logisch äquivalent: Fixiere die Strategien aller anderen gleich v_sne und prüfe, ob ein einzelner Spieler durch Abweichen gewinnen kann.
# Algorithmus:
# - Tankstelle 1 setzt einen Verkaufspreis v1
# - alle anderen (N-1) Tankstellen setzen einen Verkaufspreis vo
# - der Durchschnittspreis ist folglich (v1 + (N - 1) * vo) / N
# - wir maximieren den Nutzen U1(v1, vo) der Tankstelle 1 nach v1 (∂U1/∂v1, First-Order-Condition, FOC)
# - danach setzen wir v1 = vo = v_sne (Symmetrie) und lösen ∂U1/∂v = 0, Ergebnis ist v_sne
# Das Ergebnis v_sne ist der symmetrische Gleichgewichtspreis: kein einzelner Spieler kann durch einseitiges Abweichen von v seinen Nutzen verbessern.
def nash_equilibrium(M, N, cfg, logger, functionname="nash_equilibrium"):

    # Preise: v1 = "Abweichler" (Tankstelle 1), vo = "alle anderen Tankstellen"
    v1, vo, v = symbols("v1 vo v", real=True)

    # Durchschnittspreis bei (v1, vo,...,vo)
    v_mean = (v1 + (N - 1) * vo) / N

    # Die Gesamtnachfrage (gekaufte Menge) hängt vom Durchschnittspreis ab
    m_sum = M * (cfg.m_0 - cfg.h * (v_mean - cfg.v_0))

    # An der Tankstelle verkaufte Menge: m[n] ∝ (m_sum/N) + r * (v_mean - v[n]) mit Proportionalitätsfaktor C_mv = 1 aus C_mv * sum((m_sum/N) + r * (v_mean - v[n])) = m_sum
    m1  = (m_sum / N) + cfg.r * (v_mean - v1) # Verkaufsmenge der Tankstelle 1

    # Nutzen von Tankstelle 1
    U1 = m1 * (v1 / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.F_n - cfg.K_n

    # Nutzen U1(v1, vo) der Tankstelle 1 maximieren (First-Order-Condition, FOC: ∂U1/∂v1 = 0)
    dU1_dv1 = diff(U1, v1)
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): U1 = {U1}")
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): dU1/dv1 = {dU1_dv1}")

    # Symmetrisches Gleichgewicht: v1 = vo = v
    foc_sym = simplify(dU1_dv1.subs({v1: v, vo: v}))

    # Löse FOC im symmetrischen Punkt nach v, ∂U/∂v = 0
    v_sne = solve(Eq(foc_sym, 0), v, dict=True)
    assert(len(v_sne) > 0)
    v_sne = factor(simplify(v_sne[0][v]))

    # Verkaufsmenge m_sne und Nutzen U_sne pro Tankstelle im Gleichgewicht
    m_sne = M * (cfg.m_0 - cfg.h * (v_sne - cfg.v_0)) / N
    U_sne = m_sne * (v_sne / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.F_n - cfg.K_n
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): v_sne = {v_sne}, m_sne = {m_sne}, U_sne = {U_sne}")

    # Aufgrund des linearen Zusammenhangs zwischen Verkaufsmenge und Verkaufspreis kann die verkaufte Menge an einer besonders teuren Tankstelle rechnerisch auch negativ werden.
    # Check gegen negative Mengen:
    if m_sne.is_Number and m_sne < 0:
        logger.print(HilbertLogger.Verbosity.WARNING, f"{functionname}(M = {M}, N = {N}): m_sne = {m_sne}, non-negative value expected")
    m_sum = simplify(m_sum.subs({v1: v_sne, vo: v_sne}))
    return v_sne, m_sne, U_sne, m_sum

# Gleichgewichte für 2 Tankstellen.
# Das 2-Konkurrentenspiel (N,A,T,q,u) ist definiert durch:
# Menge der Spieler: N = { T_1 (etablierte Tankstelle), T_2 (neuer Konkurrent) }
# Menge der Aktionen: A_1 = { a (aggressiv), f (friedlich) }
# Menge der Typen: T_i = { R (Reich), A (arm) }
# Wahrscheinlichkeiten: P(T_i=R) = q, P(T_i=A) = 1 - q
# Nutzenfunktionen U_n:
#   U_n = m_n * (v_n / (1+s) - e_n) - b_n - f_n - K_n mit
#   m_n = verkaufte Menge an der n-ten Tankstelle
#   v_n = Verkaufspreis an der n-ten Tankstelle
#   s = Mehrwertsteuersatz
#   e_n = Einkaufspreis der n-ten Tankstelle
#   b_n = laufende Betriebskosten der n-ten Tankstelle
#   F_n = Investition/Abschreibung der n-ten Tankstelle, F_1 = 0, F_2 > 0
#   K_n = Kosten eines Konkurrenzkampfes, K_n(f)=0, K_n(a,T_n=R)=0, K_n(a,T_n=A)> 0
# Nebenbedingung: Ein Anbieter mit U_n < 0 macht Verlust und stellt den Betrieb ein, der Konkurrent gewinnt.
def equilibrium_two_players():
    
    logger = HilbertLogger(verbosity = HilbertLogger.Verbosity.INFO)
    M = 2 # symbols("M", int=True, positive=True) # M = Anzahl Kunden bestimmt die Nachfrage
    N = 2 # N = Anzahl Anbieter bestimmt Gleichgewichtspreis und -menge

    for symbolic in [ True, False ]:
        descr = f"(symbolisch)" if symbolic else f"(Beispiel)" 
        q = symbols("q", real=True) # q = P(T_i=R) = Wahrscheinlichkeit, dass eine Tankstelle reich ist
        cfg = HilbertParameter(symbolic = symbolic)
        v_one, m_one, U_one, m_sum_one = nash_equilibrium(M = M, N = 1, cfg = cfg, logger = logger) # Gleichgewichtspreis und -menge (M Kunden, 1 Anbieter)
        v_sne, m_sne, U_sne, m_sum_sne = nash_equilibrium(M = M, N = N, cfg = cfg, logger = logger) # Gleichgewichtspreis und -menge (M Kunden, N Anbieter)
        logger.print(HilbertLogger.Verbosity.INFO, f"Nash-Gleichgewicht bei {M} Kunden und 1 Anbietern {descr}:\n    Verkaufspreis v = {v_one}\n    Menge/Anbieter m = {m_one}\n    Nutzen/Anbieter U = {U_one}\n    Gesamtmenge = {m_sum_one}")
        logger.print(HilbertLogger.Verbosity.INFO, f"Nash-Gleichgewicht bei {M} Kunden und {N} Anbietern {descr}:\n    Verkaufspreis v = {v_sne}\n    Menge/Anbieter m = {m_sne}\n    Nutzen/Anbieter U = {U_sne}\n    Gesamtmenge = {m_sum_one}")

        # Kostenfunktionen (Investitionskosten, Kosten durch Konkurrenzkampf)
        F_1 = 0 # Tankstelle T_1 ist etabliert (keine Investitionskosten)
        F_2 = symbols("F_2", real=True) if symbolic else 1 # Konkurrent T_2 hat Investitionskosten F_2 > 0
        K_f = 0 # Friedliche Reaktion -> keine Kosten durch Konkurrenzkampf
        K_aR = 0 # Reiche Tankstellen zahlen einen Konkurrenzkampf ohne Gewinneinbuße aus ihren Rücklagen, K_n(a,T=R) = 0
        K_aA = symbols("K(a|T=A)", real=True) if symbolic else 1 # Arme Tankstellen machen Verluste durch einen Konkurrenzkampf, K_n(a,T=A) > 0
        profit_one = m_one * (v_one/(1+cfg.s) - cfg.e_n) # Gewinn pro Liter = m_sne * (v_sne/(1+cfg.s) - cfg.e_n) bei 1 Anbieter
        profit_sne = m_sne * (v_sne/(1+cfg.s) - cfg.e_n) # Gewinn pro Liter = m_sne * (v_sne/(1+cfg.s) - cfg.e_n) bei N Anbietern
        
        # Nutzenfunktionen unter der Annahme von N Anbietern (alle Anbieter ueberleben mit U_n >= 0)
        U1, U2 = {}, {} # Nutzenfunktionen: U_n = m_sne * (v_sne/(1+cfg.s) - cfg.e_n) - cfg.b_n - F_n - K_n = profit - cfg.b_n - F_n - K_n
        U1[f"f"] = profit_sne - cfg.b_n - F_1 - K_f
        U2[f"f"] = profit_sne - cfg.b_n - F_2 - K_f
        U1[f"a|T1=R,T2=R,N={N}"] = profit_sne - cfg.b_n - F_1 - K_aR
        U2[f"a|T1=R,T2=R,N={N}"] = profit_sne - cfg.b_n - F_2 - K_aR
        U1[f"a|T1=R,T2=A,N={N}"] = profit_sne - cfg.b_n - F_1 - K_aR
        U2[f"a|T1=R,T2=A,N={N}"] = profit_sne - cfg.b_n - F_2 - K_aA
        U1[f"a|T1=A,T2=R,N={N}"] = profit_sne - cfg.b_n - F_1 - K_aA
        U2[f"a|T1=A,T2=R,N={N}"] = profit_sne - cfg.b_n - F_2 - K_aR
        U1[f"a|T1=A,T2=A,N={N}"] = profit_sne - cfg.b_n - F_1 - K_aA
        U2[f"a|T1=A,T2=A,N={N}"] = profit_sne - cfg.b_n - F_2 - K_aA

        # Nutzenfunktionen unter der Annahme von 1 Anbieter (alle neuen Konkurrenten geben auf nach Konkurrenzkampf und U_n < 0)
        U1[f"a|T1=R,T2=R,N=1"] = profit_one - cfg.b_n - F_1 - K_aR
        U2[f"a|T1=R,T2=R,N=1"] = profit_one - cfg.b_n - F_2 - K_aR
        U1[f"a|T1=R,T2=A,N=1"] = profit_one - cfg.b_n - F_1 - K_aR
        U2[f"a|T1=R,T2=A,N=1"] = profit_one - cfg.b_n - F_2 - K_aA
        U1[f"a|T1=A,T2=R,N=1"] = profit_one - cfg.b_n - F_1 - K_aA
        U2[f"a|T1=A,T2=R,N=1"] = profit_one - cfg.b_n - F_2 - K_aR
        U1[f"a|T1=A,T2=A,N=1"] = profit_one - cfg.b_n - F_1 - K_aA
        U2[f"a|T1=A,T2=A,N=1"] = profit_one - cfg.b_n - F_2 - K_aA

        # Der neue Konkurrent T2 muss aufgeben, wenn er Verlust macht (U2 < 0). Dann erzielt die etablierte Tankstelle T1 den Gewinn eines einzelnen Anbieters.
        U1[f"a|T1=R,T2=R"] = Piecewise((U1[f"a|T1=R,T2=R,N=1"], U2[f"a|T1=R,T2=R,N={N}"] < 0), (U1[f"a|T1=R,T2=R,N={N}"], True))
        U1[f"a|T1=R,T2=A"] = Piecewise((U1[f"a|T1=R,T2=A,N=1"], U2[f"a|T1=R,T2=A,N={N}"] < 0), (U1[f"a|T1=R,T2=A,N={N}"], True))
        U1[f"a|T1=A,T2=R"] = Piecewise((U1[f"a|T1=A,T2=R,N=1"], U2[f"a|T1=A,T2=R,N={N}"] < 0), (U1[f"a|T1=A,T2=R,N={N}"], True))
        U1[f"a|T1=A,T2=A"] = Piecewise((U1[f"a|T1=A,T2=A,N=1"], U2[f"a|T1=A,T2=A,N={N}"] < 0), (U1[f"a|T1=A,T2=A,N={N}"], True))

        # Der etablierte Anbieter T1 muss aufgeben, wenn er Verlust macht (U1 < 0). Dann erzielt der neue Konkurrent T2 den Gewinn eines einzelnen Anbieters,
        # sofern T2 nicht vorher wg. U2 < 0 aufgegeben hat.
        U2[f"a|T1=R,T2=R"] = Piecewise((U2[f"a|T1=R,T2=R,N=1"], And(U1[f"a|T1=R,T2=R,N={N}"] < 0, U2[f"a|T1=R,T2=R,N={N}"] >= 0)), (U2[f"a|T1=R,T2=R,N={N}"], True))
        U2[f"a|T1=R,T2=A"] = Piecewise((U2[f"a|T1=R,T2=A,N=1"], And(U1[f"a|T1=R,T2=A,N={N}"] < 0, U2[f"a|T1=R,T2=A,N={N}"] >= 0)), (U2[f"a|T1=R,T2=A,N={N}"], True))
        U2[f"a|T1=A,T2=R"] = Piecewise((U2[f"a|T1=A,T2=R,N=1"], And(U1[f"a|T1=A,T2=R,N={N}"] < 0, U2[f"a|T1=A,T2=R,N={N}"] >= 0)), (U2[f"a|T1=A,T2=R,N={N}"], True))
        U2[f"a|T1=A,T2=A"] = Piecewise((U2[f"a|T1=A,T2=A,N=1"], And(U1[f"a|T1=A,T2=A,N={N}"] < 0, U2[f"a|T1=A,T2=A,N={N}"] >= 0)), (U2[f"a|T1=A,T2=A,N={N}"], True))

        # Erwartungsnutzen U1 der Tankstelle T1 bei aggressiver Reaktion
        U1[f"a|T1=R"] = q * U1[f"a|T1=R,T2=R"] + (1 - q) * U1[f"a|T1=R,T2=A"]
        U1[f"a|T1=A"] = q * U1[f"a|T1=A,T2=R"] + (1 - q) * U1[f"a|T1=A,T2=A"]

        # Erwartungsnutzen U2 der Tankstelle T2 bei aggressiver Reaktion
        U2[f"a|T2=R"] = q * U2[f"a|T1=R,T2=R"] + (1 - q) * U2[f"a|T1=A,T2=R"]
        U2[f"a|T2=A"] = q * U2[f"a|T1=R,T2=A"] + (1 - q) * U2[f"a|T1=A,T2=A"]

        # Strategie 1: T1 wählt immer friedlich
        equilibrium_satisfiable(f"T1 waehlt friedlich", And(U1["f"] >= U1["a|T1=R"], U1["f"] >= U1["a|T1=A"]), symbolic, q, logger)

        # Strategie 2: T1 wählt immer aggressiv
        equilibrium_satisfiable(f"T1 waehlt aggressiv", And(U1["a|T1=R"] >= U1["f"], U1["a|T1=A"] >= U1["f"]), symbolic, q, logger)

        # Strategie 3: T1=R wählt immer a, T1=A wählt immer f
        equilibrium_satisfiable(f"T1=R waehlt a, T1=A waehlt f", And(U1["a|T1=R"] >= U1["f"], U1["f"] >= U1["a|T1=A"]), symbolic, q, logger)

        # Strategie 4: T1=R wählt immer f, T1=A wählt immer a
        equilibrium_satisfiable(f"T1=R waehlt f, T1=A waehlt a", And(U1["f"] >= U1["a|T1=R"], U1["a|T1=A"] >= U1["f"]), symbolic, q, logger)

        logger.print(HilbertLogger.Verbosity.INFO, f"")

if __name__ == "__main__":
    equilibrium_two_players() # Der allereinfachste Fall: M Kunden, eine etablierte Tankstelle und Eintritt eines einzigen neuen Konkurrenten (2-Konkurrenten-Spiel)
