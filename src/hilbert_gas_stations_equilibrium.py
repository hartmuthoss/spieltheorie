"""
Gleichgewichte in Hilbert-Tankstellen, Tankstellen-Setup mit M Kunden, N etablierten Tankstellen und L neuen Konkurrenten.

Es gibt 4 Gruppen von Tankstellen: T1R (reiche etablierte Tankstellen), T1A (arme etablierte Tankstellen), T2R (reiche neue Konkurrenten), T2A (arme neue Konkurrenten)
Das (N,L)-Konkurrentenspiel ((N,L),A,T,q,u) ist definiert durch:
Menge der Spieler: { T1 (N etablierte Tankstelle), T2 (L neue Konkurrent) }
Menge der Aktionen: A1 = { a (aggressiv), f (friedlich) }
Menge der Typen: Ti = { R (Reich), A (arm) }
Wahrscheinlichkeiten: P(Ti=R) = q, P(Ti=A) = 1 - q
Nutzenfunktionen U_n:
  U_n = m_n * (v_n / (1+s) - e_n) - b_n - f_n - K_n mit
  m_n = verkaufte Menge an der n-ten Tankstelle
  v_n = Verkaufspreis an der n-ten Tankstelle
  s = Mehrwertsteuersatz
  e_n = Einkaufspreis der n-ten Tankstelle
  b_n = laufende Betriebskosten der n-ten Tankstelle
  F_n = Investition/Abschreibung der n-ten Tankstelle, F_1 = 0, F_2 > 0
  K_n = Kosten eines Konkurrenzkampfes, K_n(f)=0, K_n(a,T_n=R)=0, K_n(a,T_n=A)> 0
Nebenbedingung: Ein Anbieter mit U_n < 0 macht Verlust und stellt den Betrieb ein, der Konkurrent gewinnt.

Beschreibung und Erlaeuterungen in ../docs/HilbertTankstellen.md
"""
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from enum import IntEnum
from pathlib import Path
from sympy import *
from sympy.assumptions.relation.binrel import AppliedBinaryRelation
from sympy.core.relational import Relational

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

# Intervall für Verkaufspreis v_sne und Verkaufsmenge m_sne inkl. Beispielwert 
# (z.B. Verkaufspreis bei q = 0.5 oder q = Intervallmitte)
class VMInterval():
    def __init__(self, name, interval = S.EmptySet, example = None):
        self.name = name
        self.interval = interval
        self.example = example
    def __str__(self):
        if self.interval.sup - self.interval.inf < 1.0e-6:
            return f"{self.name} {self.example:.4f}"
        return f"{self.name} {self.example:.4f} (min:{self.interval.inf:.4f}, max:{self.interval.sup:.4f})"

# Erfüllbarkeit einer Gleichgewichtsbedingung, Ergebnis von equilibrium_satisfiable.
# Eine Gleichgewichtsbedingung ist erfüllbar, wenn:
# - status True ist und q den Beispielwert aus dem grid-check hat, oder
# - status True ist und q im Intervall liegt.
class SatisfiableResult():
    def __init__(self, status = False, grid_q = math.nan, q_interval = Interval(0, -1)):
        self.status = status
        self.grid_q = grid_q
        self.q_interval = q_interval

# Hilbert-Tankstellen bestehen aus 4 Gruppen mit unterschiedlichen Nutzenfunktionen U_g:
# T1R (reiche etablierte Tankstellen), T1A (arme etablierte Tankstellen), T2R (reiche neue Konkurrenten), T2A (arme neue Konkurrenten)
# Wenn die Tankstellen einer Gruppe Verluste schreiben (U_g < 0), dann stellen die Tankstellen ihren Betrieb ein und die Gruppe wird gelöscht.
class HilbertGroup():

    # Parameter einer Tankstellengruppe
    def __init__(self, name, symbolic, is_established, is_rich, logger = HilbertLogger(verbosity = HilbertLogger.Verbosity.INFO)):
        self.name = name           # Name der Gruppe, z.B. "T1R"
        self.symbolic = symbolic   # symbolisch oder numerisch mit Beispielswerten
        self.logger = logger       # logging
        self.is_established = is_established # etablierte Tankstelle oder neuer Konkurrent
        self.is_rich = is_rich     # arme oder reiche Tankstelle
        self.cfg = HilbertParameter(symbolic, is_established, is_rich) # Parameter (symbolisch oder konkrete Beispielwerte), etablierte Tankstelle oder neuer Konkurrent, arme oder reiche Tankstelle
        self.setup(None, None, None)

    # Setup Anzahl Tankstellen der Gruppe, Gleichgewichtspreis und -menge (abhängig von der Gesamtanzahl an Anbietern und Kunden) und der Nutzenfunktionen
    def setup(self, N, v_sne, m_sne):
        self.N = N         # Anzahl Tankstellen der Gruppe, z.B. q*N in Gruppe "T1R" mit N = Anzahl etablierter Tankstellen und q = P(Ti=R)
        self.v_sne = v_sne # Gleichgewichtspreis (Verkaufspreis, abhängig von der Gesamtanzahl an Anbietern)
        self.m_sne = m_sne # Gleichgewichtsmenge (Verkaufsmenge pro Anbieter, abhängig von der Gesamtanzahl an Anbietern und Gesamtanzahl an Kunden)
        # Nutzenfunktion: U = m_sne * (v_sne/(1+cfg.s) - cfg.e) - cfg.b - cfg.F - K
        self.U = {}
        if self.N is not None and self.v_sne is not None and self.m_sne is not None:
            profit_sne = self.m_sne * (self.v_sne / (1 + self.cfg.s) - self.cfg.e_n) # Gewinn pro Liter = m_sne * (v_sne/(1+cfg.s) - cfg.e_n)
            self.U["f"] = profit_sne - self.cfg.b_n - self.cfg.F_n - 0               # Friedliche Reaktion -> keine Kosten durch Konkurrenzkampf, K_n = 0
            self.U["a"] = profit_sne - self.cfg.b_n - self.cfg.F_n - self.cfg.K_n    # Aggressive Reaktion -> Kosten K_n durch Konkurrenzkampf
            # Anmerkung zu den "Aggressionskosten" K_n: Arme Tankstellen (etablierte und neue) haben Aggressionskosten (Finanzierung per teurem Kredit, K_n > 0), 
            # reiche Tankstellen dagegen nicht (Finanzierung kostenlos aus Rücklagen, K_n = 0). Sobald eine etablierte Tankstelle (T1) aggressiv spielt, wird U["a"] benutzt, 
            # und damit fällt K_n>0 bei allen armen Tankstellen an, während für reiche Tankstellen U["a"] = U["f"] und K_n = 0 ist. 
            # Neue Konkurrenten (T2) haben zwar keine Aggressionsoption, arme Neukonkurrenten zahlen aber genauso wie arme Etablierte im Falle Konkurrenzkampf.

# Die Menge aller Tankstellengruppen
class HilbertGroups():

    # Ergebnis von group_equilibrium(): Gefundene Gleichgewichte inkl. Strategie, aktive Gruppen, Gleichgewichtspreis und -menge
    class Equilibrium():
        # Initialisierung
        def __init__(self, status, strategy, groups, q_condition, v, m): # True, strategy, groups, q_condition, grid_v, grid_m)
            self.status = status           # True (Gleichgewicht existiert) oder False (falls U_g < 0 oder N_g <= 0)
            self.strategy = strategy       # Strategie: "T1f" (immer friedlich), "T1a" (immer aggressiv), "T1fa" (T1=R friedlich, T1=A aggressiv), "T1af" (T1=R aggressiv, T1=A friedlich)
            self.groups = groups           # Liste aktiver Gruppen: "T1R" (reiche etablierte Tankstellen), "T1A" (arme etablierte Tankstellen), "T2R" (reiche neue Konkurrenten), "T2A" (arme neue Konkurrenten)
            self.q_condition = q_condition # Wertebereich für q = P(T_i=R), Intervall oder Bedingung
            self.v = v                     # Gleichgewichtspreis (Verkaufspreis)
            self.m = m                     # Gleichgewichtsmenge pro Anbieter
        # string-Serialisierung
        def __str__(self):
            status_str = "(Gleichgewicht)" if self.status else "(Gleichgewicht UNBEKANNT)"
            q_str = f"q in {self.q_condition}" if isinstance(self.q_condition, FiniteSet) else f"{tostr(self.q_condition)}"
            return f"Strategie {self.strategy} {status_str}: {len(self.groups)} aktive Gruppen {tostr(self.groups)}, {q_str}, {self.v}, {self.m}"

    # Konstruktor
    def __init__(self, symbolic = False, logger = HilbertLogger(verbosity = HilbertLogger.Verbosity.INFO)):
        self.symbolic = symbolic # symbolisch oder numerisch mit Beispielswerten
        self.logger = logger     # logging

    # Bestimmung der Gleichgewichte: 
    # 1. Bestimmung Gleichgewichtspreis und -menge, abhängig von N = Anzahl etablierter Tankstellen, LNR = L/N, L = LNR*N = Anzahl neuer Konkurrenten, MNLR = M/(N+L), M = MNLR*(N+L) = MNLR*(N+LNR*N) = Anzahl Kunden
    # 2. Prüfen der Gleichgewichtsbedingungen für 4 unterschiedliche Strategien (T1 wählt immer friedlich, T1 wählt immer aggressiv, T1=R friedlich und T1=A aggressiv, oder T1=R aggressiv und T1=A friedlich)
    # 3. Falls Gruppe mit Nutzen U_g < 0 existiert: Schlechteste Gruppe entfernen und wiederholen
    def group_equilibrium(self, N, LNR, MNLR = 1):

        # Initialisierung der Tankstellengruppen, Strategien und Bereiche für Wahrscheinlichkeit q = P(T_i=R)
        groups = {
            "T1R": HilbertGroup("T1R", self.symbolic, True, True, self.logger),   # Gruppe T1R: reiche etablierte Tankstellen
            "T1A": HilbertGroup("T1A", self.symbolic, True, False, self.logger),  # Gruppe T1A: arme etablierte Tankstellen
            "T2R": HilbertGroup("T2R", self.symbolic, False, True, self.logger),  # Gruppe T2R: reiche neue Konkurrenten
            "T2A": HilbertGroup("T2A", self.symbolic, False, False, self.logger), # Gruppe T2A: arme neue Konkurrenten
            }
        strategies = ("T1f",  # Etablierte Tankstellen T1 wählen immer friedlich
                      "T1a",  # Etablierte Tankstellen T1 wählen immer aggressiv
                      "T1fa", # Etablierte reiche Tankstellen T1=R wählen friedlich, arme T1=A wählen aggressiv 
                      "T1af") # Etablierte reiche Tankstellen T1=R wählen aggressiv, arme T1=A wählen friedlich 
        q = symbols("q", real=True)     # q = P(Ti=R) = Wahrscheinlichkeit, dass eine Tankstelle reich ist
        q_condition = And(0 < q, q < 1) # 0 < q < 1

        # Rekursive Suche nach Gleichgewichten mit 0 < q < 1
        results = {}
        results = self.group_equilibrium_recursive(num_N = N, LNR = LNR, MNLR = MNLR, groups = groups, strategies = strategies, results = results, q = q, q_condition = q_condition)
        if not self.symbolic:
            for strategy, result in results.items():
                for r in result:
                    self.logger.print(HilbertLogger.Verbosity.INFO, f"Ergebnis für N={N}, L/N={LNR}, L={LNR*N}, M/(N+L)={MNLR}, M={MNLR*(N+LNR*N)}, {r}")

            # Gleichgewichte in den Randfällen q = 0 und q = 1
            special_cases = [({"T1A": groups["T1A"], "T2A": groups["T2A"] }, ("T1f", "T1a"), 0), # q=0: special_cases[0] = (groups=("T1A","T2A"), strategies=("T1f","T1a"), q=0)
                             ({"T1R": groups["T1R"], "T2R": groups["T2R"] }, ("T1f", "T1a"), 1)] # q=1: special_cases[1] = (groups=("T1R","T2R"), strategies=("T1f","T1a"), q=1) 
            special_results = [{}, {}]
            for i, case in enumerate(special_cases):
                special_results[i] = self.group_equilibrium_recursive(num_N = N, LNR = LNR, MNLR = MNLR, groups = case[0], strategies = case[1], results = special_results[i], q = case[2], q_condition = S.true)
                for strategy, result in special_results[i].items():
                    for r in result:
                        if strategy not in results:
                            results[strategy] = []
                        results[strategy].append(r)
                        self.logger.print(HilbertLogger.Verbosity.INFO, f"Ergebnis für N={N}, L/N={LNR}, L={LNR*N}, M/(N+L)={MNLR}, M={MNLR*(N+LNR*N)}, q={case[2]}, {r}")

        self.logger.print(HilbertLogger.Verbosity.INFO, f"")
        return results
       
    # Rekursive Suche nach Gleichgewichten: 
    # 1. Bestimmung Gleichgewichtspreis und -menge, abhängig von num_N = Anzahl etablierter Tankstellen, LNR = L/N, L = LNR*N = Anzahl neuer Konkurrenten, MNLR = M/(N+L), M = MNLR*(N+L) = MNLR*(N+LNR*N) = Anzahl Kunden
    # 2. Prüfen der Gleichgewichtsbedingungen für 4 unterschiedliche Strategien (T1 wählt immer friedlich, T1 wählt immer aggressiv, T1=R friedlich und T1=A aggressiv, oder T1=R aggressiv und T1=A friedlich)
    # 3. Falls Gruppe mit Nutzen U_g < 0 existiert: Schlechteste Gruppe entfernen und wiederholen
    def group_equilibrium_recursive(self, num_N, LNR, MNLR, groups, strategies, results, q, q_condition):

        # Anzahl Tankstellen in jeder Gruppe, abhängig davon welche Gruppen existieren
        # q = symbols("q", real=True) # q = P(Ti=R) = Wahrscheinlichkeit, dass eine Tankstelle reich ist
        N = symbols("N", int=True, positive=True) if num_N == oo else num_N # N = Anzahl etablierte Tankstellen
        N_g = {}                                                 # Anzahl aktive Tankstellen je Gruppe
        N_g["T1R"] = q * N if "T1R" in groups else 0             # Anzahl aktive Tankstellen in Gruppe T1R (reiche etablierte Tankstellen)
        N_g["T1A"] = (1 - q) * N if "T1A" in groups else 0       # Anzahl aktive Tankstellen in Gruppe T1A (arme etablierte Tankstellen)
        N_g["T2R"] = q * LNR * N if "T2R" in groups else 0       # Anzahl aktive Tankstellen in Gruppe T2R (reiche neue Konkurrenten)
        N_g["T2A"] = (1 - q) * LNR * N if "T2A" in groups else 0 # Anzahl aktive Tankstellen in Gruppe T2A (arme neue Konkurrenten)
        
        # Gleichgewichtspreis und -menge in Abhängigkeit von Anzahl aktiver Tankstellen und Kunden
        N_active = N_g["T1R"] + N_g["T1A"]
        L_active = N_g["T2R"] + N_g["T2A"]
        MNR_active = MNLR * (N + LNR * N) / (N_active + L_active) # MNLR = M/(N+L), MNR_active = M/(N_active+L_active) = MNLR*(N+L)/(N_active+L_active) = MNLR*(N+LNR*N)/(N_active+L_active)
        v_sne, m_sne, _, _ = nash_equilibrium(N_active + L_active, MNR_active, HilbertParameter(self.symbolic), self.logger) # Gleichgewichtspreis und -menge
        if num_N == oo:
            v_sne = limit(v_sne, N, num_N)
            m_sne = limit(m_sne, N, num_N)
            # N_g["T1R"] = limit(N_g["T1R"], N, num_N) if N_g["T1R"] != 0 else N_g["T1R"]
            # N_g["T1A"] = limit(N_g["T1A"], N, num_N) if N_g["T1A"] != 0 else N_g["T1A"]
            # N_g["T2R"] = limit(N_g["T2R"], N, num_N) if N_g["T2R"] != 0 else N_g["T2R"]
            # N_g["T2A"] = limit(N_g["T2A"], N, num_N) if N_g["T2A"] != 0 else N_g["T2A"]
        self.logger.print(HilbertLogger.Verbosity.INFO, f"Hilbert-Tankstellen: N={num_N}, L/N={LNR}, L={LNR*num_N}, M/(N+L)={MNLR}, M={MNLR*(num_N+LNR*num_N)}, aktive Gruppen: {tostr(groups)}, aktive Strategien: {strategies}, Nash-Gleichgewicht: v_sne={tostr(v_sne)}, m_sne={tostr(m_sne)}")
        
        # Setup der Nutzenfunktionen in jeder Gruppe
        for name, group in groups.items():
            assert_or_die(isinstance(q_condition, And) or q_condition == S.true or q_condition.contains(q), f"assert(isinstance({q_condition}, And) or {q_condition} == S.true or {q_condition}.contains({q}))", self.logger)
            N_g_pos_satisfiable = reduce_inequalities([N_g[name] > 0] + list(q_condition.args), q) # Ist (N_g[name] > 0 & q_condition) erfüllbar?
            if num_N == oo and N_g_pos_satisfiable.has(N) and not N_g_pos_satisfiable == S.false:  # mit limit falls N -> oo
                N_g_pos_satisfiable = limit(N_g_pos_satisfiable, N, num_N)
            if not N_g_pos_satisfiable == S.false: # (N_g[name] > 0 & q_condition) ist nicht unerfüllbar, Gruppe kann mehr als 0 Anbieter haben
                group.setup(N_g[name], v_sne, m_sne)
                self.logger.print(HilbertLogger.Verbosity.INFO, f"HilbertGroup({group.name}): U['f']={tostr(group.U['f'])}, U['a']={tostr(group.U['a'])}, N={group.N}")

        # Strategie: T1=R und T1=A wählen gleich, d.h. T1 wählt immer friedlich oder T1 wählt immer aggressiv
        equilibrium_status = { } # Für jede Strategie: Gleichgewicht existiert oder existiert nicht, evtl. mit Beispielbelegung und Intervall für q
        if "T1R" in groups or "T1A" in groups:
            condition_T1_f = True
            condition_T1_a = True
            if "T1R" in groups:
                condition_T1_f = And(condition_T1_f, groups["T1R"].U["f"] >= groups["T1R"].U["a"])
                condition_T1_a = And(condition_T1_a, groups["T1R"].U["a"] >= groups["T1R"].U["f"])
            if "T1A" in groups:
                condition_T1_f = And(condition_T1_f, groups["T1A"].U["f"] >= groups["T1A"].U["a"])
                condition_T1_a = And(condition_T1_a, groups["T1A"].U["a"] >= groups["T1A"].U["f"])
            if "T1f" in strategies:
                equilibrium_status["T1f"] = equilibrium_satisfiable(f"T1 wählt friedlich", condition_T1_f, self.symbolic, q, q_condition, self.logger)
            if "T1a" in strategies:
                equilibrium_status["T1a"] = equilibrium_satisfiable(f"T1 wählt aggressiv", condition_T1_a, self.symbolic, q, q_condition, self.logger)

        # Strategie: T1=R und T1=A wählen unterschiedlich, d.h. T1=R wählt friedlich, T1=A wählt aggressiv, oder umgekehrt
        if "T1R" in groups and "T1A" in groups:
            condition_T1_fa = And(groups["T1R"].U["f"] >= groups["T1R"].U["a"], groups["T1A"].U["a"] >= groups["T1A"].U["f"])
            condition_T1_af = And(groups["T1R"].U["a"] >= groups["T1R"].U["f"], groups["T1A"].U["f"] >= groups["T1A"].U["a"])
            if "T1fa" in strategies:
                equilibrium_status["T1fa"] = equilibrium_satisfiable(f"T1=R friedlich, T1=A aggressiv", condition_T1_fa, self.symbolic, q, q_condition, self.logger)
            if "T1af" in strategies:
                equilibrium_status["T1af"] = equilibrium_satisfiable(f"T1=R aggressiv, T1=A friedlich", condition_T1_af, self.symbolic, q, q_condition, self.logger)

        # Gruppen bzw. Tankstellen mit U_g < 0 überleben nicht und werden geschlossen. Die schlechteste Gruppe mit U_g < 0 wird entfernt.
        if not self.symbolic: # U_g < 0 nur im numerischen Fall
            for strategy in strategies:
                if strategy in equilibrium_status and equilibrium_status[strategy].status is True:
                    payoff_results = self.evaluate_groups_payoff(strategy, groups, "f" in strategy, "a" in strategy, q, equilibrium_status[strategy], v_sne, m_sne)
                    for result in payoff_results:
                        if result.status is True: # Erfolg: Gleichgewicht gefunden und alle U_g >= 0
                            if strategy not in results:
                                results[strategy] = []
                            results[strategy].append(result)
                            self.logger.print(HilbertLogger.Verbosity.INFO, f"{result}")
                        else: # Schlechteste Gruppe mit U_g < 0 entfernt -> wiederholen
                            results = self.group_equilibrium_recursive(num_N = num_N, LNR = LNR, MNLR = MNLR, groups = result.groups, strategies = (result.strategy,), results = results, q = q, q_condition = result.q_condition)
        return results

    # Teilt gegebene Tankstellen-Gruppen nach U_g >= 0 (d.h. Erfolg, Gleichgewicht existiert für eine Strategie und einen Wertebereich q) 
    # und U_g < 0 (d.h. schlechteste Gruppe entfernen und group_equilibrium_recursive wiederholen)
    def evaluate_groups_payoff(self, strategy, groups, f_onpath, a_onpath, q, equilibrium_status, v_sne, m_sne):
        
        # (Name, Nutzenfunktion) aller Gruppen
        g_Us = [ (g_name, groups[g_name].U["f"]) for g_name in groups.keys() if f_onpath ] + [ (g_name, groups[g_name].U["a"]) for g_name in groups.keys() if a_onpath ] 
        if len(g_Us) <= 0:
            return []
        
        # Gleichgewichtspreis und -menge im Erfolgsfall
        grid_q = equilibrium_status.grid_q
        grid_v, grid_m = interval_v_m(v_sne, m_sne, q, equilibrium_status.grid_q, equilibrium_status.q_interval, self.logger)
        
        # Evaluierung (U_g >= 0): Ist die Bedingung (U_g >= 0) für alle g immer oder niemals erfüllbar?
        q_condition = interval_to_condition(q, equilibrium_status.q_interval)
        assert_or_die(isinstance(q_condition, And) or q_condition == S.true or q_condition.contains(q), f"assert(isinstance({q_condition}, And) or {q_condition} == S.true or {q_condition}.contains({q}))", self.logger)
        g_U_pos_list = [ g_U[1] >= 0 for g_U in g_Us ] # UND-Argumente: alle U_g >= 0
        g_U_pos_sol = reduce_inequalities(g_U_pos_list, q)
        if g_U_pos_sol == S.true: # Erfolg: Alle U_g >= 0 ist immer erfüllt
            return [ HilbertGroups.Equilibrium(True, strategy, groups, q_condition, grid_v, grid_m) ]
        if g_U_pos_sol == S.false: # Kein Erfolg: Alle U_g >= 0 ist niemals erfüllt => Schlechteste Gruppe entfernen und wiederholen
            g_U_list = [ (g_U[0], g_U[1].subs(q,grid_q)) for g_U in g_Us ]  # Liste aller U_g(q=grid_q)
            worst_groups = sorted(g_U_list, key=lambda g_U: g_U[1])         # sortiert nach U_g
            worst_U = worst_groups[0][1] + np.finfo(np.float32).eps
            worst_groups = [ g[0] for g in worst_groups if g[1] <= worst_U ] # Falls mehrere Gruppen gleich schlechte Nutzen haben:
            worst_groups = [ g for g in worst_groups if "A" in g ] + [ g for g in worst_groups if not "A" in g ] # dann zuerst arme Tankstellen entfernen
            remaining_groups = groups.copy()
            remaining_groups.pop(worst_groups[0], None)
            if len(remaining_groups) > 0:
                self.logger.print(HilbertLogger.Verbosity.INFO, f"Strategie {strategy}: Gruppe {worst_groups[0]} mit U_g={tostr(worst_U)} wird entfernt")
                return [ HilbertGroups.Equilibrium(False, strategy, remaining_groups, q_condition, grid_v, grid_m) ]

        # Evaluierung (U_g >= 0 für alle g UND q_condition): Ist (U_g >= 0) für alle g erfüllbar in Abhängigkeit von unterschiedlichen Wertebereichen q ∈ Q_g?
        # Dann Gruppen aufteilen:
        # - U_g >= 0 erfüllbar für alle g mit q ∈ Q_g
        # - schlechteste Gruppe entfernen und wiederholen mit q ∉ Q_g
        # Anm.: An dieser Stelle kann reduce_inequalities mächtig lange Ausdrücke für q mit sehr vielen UND/ODER-verknüpften Termen liefern.
        # Pragmatische Lösung: Die tatsächliche Erfüllbarkeit des von reduce_inequalities ermittelten Audruck für q wird mit grid auf konkrete Werte getestet
        # und damit ein Intervall für q approximiert. Das gefundene Intervall kann zu klein sein oder könnte kleine Bereiche überdecken, die die Bedingung nicht erfüllen.
        results = []
        g_U_pos_sol = reduce_inequalities(g_U_pos_list + list(q_condition.args), q) # Wertebereich q mit U_g >= 0
        # Allgemeine Lösung: Liste über Intervalle, die g_U_pos_sol erfüllen
        g_U_pos_sol_intervals = search_for_q_intervals(g_U_pos_sol, q, equilibrium_status.q_interval, self.logger) 
        for q_interval in g_U_pos_sol_intervals:
            g_U_pos_q = Rational(q_interval.inf + q_interval.sup, 2)
            grid_v, grid_m = interval_v_m(v_sne, m_sne, q, Rational(q_interval.inf + q_interval.sup, 2), q_interval, self.logger)
            result = HilbertGroups.Equilibrium(True, strategy, groups, interval_to_condition(q, q_interval), grid_v, grid_m)
            results.append(result)
        # Numerische Näherung per grid (beispielhaftes Teilintervall, welches g_U_pos_sol erfüllt, aber nicht die gesamte Lösung abdeckt):
        # g_U_pos_sat, g_U_pos_q = probability_grid_check(g_U_pos_sol, q)
        # if g_U_pos_sat == True and equilibrium_status.q_interval.contains(g_U_pos_q):
        #     q_interval = probability_grid_interval(g_U_pos_sol, q, g_U_pos_q) #
        #     grid_v = simplify(v_sne.subs(q, g_U_pos_q))
        #     grid_m = simplify(m_sne.subs(q, g_U_pos_q))
        #     result = HilbertGroups.Equilibrium(True, strategy, groups, simplify(And(q_condition, q >= q_interval.inf, q <= q_interval.sup)), grid_v, grid_m)
        #     results.append(result)

        # Evaluierung (U_g < 0 UND q_condition): Existiert ein Wertebereich q ∈ Q_g mit einer oder mehrere Gruppen mit U_g < 0? Dann schlechteste Gruppe entfernen und wiederholen mit q ∈ Q_g
        worst_group = (None, 0, None, 0, None) # (group_name, U_g, condition, grid_q, q_interval) der schlechtesten Gruppe
        for g_U in reversed(g_Us):
            g_U_neg_sol = reduce_inequalities([g_U[1] < 0] + list(q_condition.args), q)
            # Allgemeine Lösung: Liste über Intervalle, die g_U_neg_sol erfüllen
            g_U_neg_sol_intervals = search_for_q_intervals(g_U_neg_sol, q, equilibrium_status.q_interval, self.logger)
            for q_interval in g_U_neg_sol_intervals:
                g_U_neg_q = Rational(q_interval.inf + q_interval.sup, 2)
                q_min = q_interval.inf + Rational(1,1000000) if q_interval.is_Interval and q_interval.left_open else q_interval.inf
                q_max = q_interval.sup - Rational(1,1000000) if q_interval.is_Interval and q_interval.right_open else q_interval.sup
                U = min(g_U[1].subs(q, g_U_neg_q), g_U[1].subs(q, q_min), g_U[1].subs(q, q_max))
                # U_g(q) = g_U[1](q) kann Extrema haben. Für die min und max-Werte von U_g also dU_g(q)/dq = 0 mit Randbedingung q innerhalb q_interval berechnen:
                if g_U[1].has(q):
                    dU_dq = diff(g_U[1], q)
                    q_0_sol = solve(Eq(dU_dq, 0), q)
                    for q_0 in q_0_sol:
                        if q_0.is_real and q_interval.contains(q_0):
                            U = min(U, g_U[1].subs(q, q_0))
                if U < worst_group[1]:
                    worst_group = (g_U[0], U, g_U_neg_sol, g_U_neg_q, q_interval)
                elif U == worst_group[1] and "A" in g_U[0] and "A" not in worst_group[0]: # Bei 2 gleich schlechten Gruppen zuerst die armen Tankstellen entfernen
                    worst_group = (g_U[0], U, g_U_neg_sol, g_U_neg_q, q_interval)
            # Numerische Näherung per grid (beispielhaftes Teilintervall, welches g_U_neg_sol erfüllt, aber nicht die gesamte Lösung abdeckt):
            # g_U_neg_sat, g_U_neg_q = probability_grid_check(g_U_neg_sol, q)
            # if g_U_neg_sat == True:
            #     U = g_U[1].subs(q, g_U_neg_q)
            #     if U < worst_group[1]:
            #         worst_group = (g_U[0], U, g_U_neg_sol, g_U_neg_q)
            #     elif U == worst_group[1] and "A" in g_U[0] and "A" not in worst_group[0]: # Bei 2 gleich schlechten Gruppen zuerst die armen Tankstellen entfernen
            #         worst_group = (g_U[0], U, g_U_neg_sol, g_U_neg_q)
        if worst_group[0] is not None:
            remaining_groups = groups.copy()
            remaining_groups.pop(worst_group[0], None)
            if len(remaining_groups) > 0:
                q_interval = worst_group[4]
                grid_v, grid_m = interval_v_m(v_sne, m_sne, q, worst_group[3], q_interval, self.logger)
                result = HilbertGroups.Equilibrium(False, strategy, remaining_groups, interval_to_condition(q, q_interval), grid_v, grid_m)
                results.append(result)
                self.logger.print(HilbertLogger.Verbosity.INFO, f"Strategie {strategy}: Gruppe {worst_group[0]} mit U_g={tostr(worst_group[1])} wird entfernt")
        return results

# assert and raise exception
def assert_or_die(assert_cond, msg, logger):
    if not assert_cond:
        logger.print(HilbertLogger.Verbosity.ERROR, f"## ERROR {__file__}: {assert_cond} failed, {msg}")
        raise(Exception(f"{assert_cond} failed, {msg}"))
    assert(assert_cond)

# Konvertierung in string mit 4 Nachkommastellen, falls expr vom Typ Float ist
def tostr(expr):
    if isinstance(expr,dict):
        return tuple( key for key in expr )
    if isinstance(expr, Float):
        return f"{expr:.4f}"
    if isinstance(expr, Expr) and expr.is_Float:
        return f"{expr:.4f}"
    if isinstance(expr, Basic):
        return f"{expr.xreplace({n : round(n, 4) for n in expr.atoms(Number)})}"
    return f"{expr}"

# Konvertiert x in json-serializable Objekt inkl. Infinity, Rational, Equilibrium, etc.
def tojson(x):
    if isinstance(x, Basic):
        return srepr(x)
    if isinstance(x, HilbertGroups.Equilibrium):
        return (x.strategy, x.status, tojson(x.groups), srepr(x.q_condition), srepr(x.v.example), srepr(x.v.interval), srepr(x.m.example), srepr(x.m.interval))
    if isinstance(x, HilbertGroup):
        return (x.name, srepr(x.N), srepr(x.v_sne), srepr(x.m_sne), srepr(x.U["f"]), srepr(x.U["a"]))
    if isinstance(x, tuple):
        return tuple(tojson(y) for y in x)
    if isinstance(x, list):
        return [ tojson(y) for y in x ]
    if isinstance(x, dict):
        return { tojson(key): tojson(value) for key, value in x.items() }
    return x
        
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

# Konvertiert ein Interval in eine AND-Bedingung
def interval_to_condition(q, q_interval):
    if isinstance(q_interval, Interval):
        return And(
            q > q_interval.inf if q_interval.left_open else q >= q_interval.inf, 
            q < q_interval.sup if q_interval.right_open else q <= q_interval.sup)
    return q_interval

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
def search_for_q_intervals(q_condition, q, q_interval, logger):
    if q_condition == S.true: # Bedingung immer erfüllbar
        return [ q_interval ]
    elif q_condition == S.false: # Bedingung niemals erfüllbar
        return []
    else:
        assert_or_die(isinstance(q_condition, And) or q_condition == S.true or q_condition.contains(q), f"assert(isinstance({q_condition}, And) or {q_condition} == S.true or {q_condition}.contains({q}))", logger)
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
            assert_or_die(False, f"solveset_recursive returned {q_sol} with unknown type {type(q_sol)}", logger)
            return [] 
    return []

# Sucht das kleinste Intervall für q, das eine gegebene UND-Verknüpfung von Relationen erfüllt
def intersect_q_intervals(condition, q, logger):
    assert_or_die(isinstance(condition, And) or condition == S.true or condition.contains(q), f"assert(isinstance({condition}, And) or {condition} == S.true or {condition}.contains({q}))", logger)
    if (q == 0 or q == 1) and (condition == S.true or condition.contains(q)):
        return Interval(q, q)
    q_interval = Interval.open(0, 1)
    q_terms = [ term for term in condition.args if (term.lhs.has(q) or term.rhs.has(q)) ] # alle Terme die q enthalten
    for term in q_terms:
        interval = solve_univariate_inequality(term, q, relational=False)
        q_interval = q_interval.intersect(interval)
    return q_interval

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
        if probability_val_check(se_condition, q, grid_q) == True:
            return True, grid_q
    return False, math.nan

# Ermittelt ein Interval [ grid_q - q1, grid_q + q2 ], welches se_condition erfüllt.
# q1, q2 durch schrittweises Testen von se_condition mit fester Schrittlänge grid_step.
def probability_grid_interval(self, se_condition, q, grid_q, grid_step = 0.01):
    q_interval = Interval(grid_q, grid_q)
    for q_tst in np.arange(grid_q, 0, -grid_step):
        if probability_val_check(se_condition, q, q_tst) == True:
            q_interval = Interval(q_tst, q_interval.sup)
    for q_tst in np.arange(grid_q, 1, grid_step):
        if probability_val_check(se_condition, q, q_tst) == True:
            q_interval = Interval(q_interval.inf, q_tst)
    return q_interval

# Berechnet die Intervalle für Verkaufspreis und -menge aus v_sne(q) und m_sne(q) für ein gegebenes q-Intevall
def interval_v_m(v_sne, m_sne, q, grid_q, q_interval, logger):
    if q == 0 or q == 1:
        assert_or_die(v_sne.has(q) == False and m_sne.has(q) == False, f"interval_v_m(v_sne={v_sne}, m_sne={m_sne}, q={q}, grid_q={grid_q}, q_interval={q_interval}): v_sne.has(q)={v_sne.has(q)}, m_sne.has(q)={m_sne.has(q)}, should be False", logger)
        return [ VMInterval("Verkaufspreis", interval = Interval(v_sne, v_sne), example =v_sne), VMInterval("Verkaufsmenge/Anbieter", interval = Interval(m_sne, m_sne), example = m_sne) ]
    q_min = q_interval.inf + Rational(1,1000000) if q_interval.left_open else q_interval.inf  # für q_interval.inf = 0 kann eine Division durch 0 zu nan führen
    q_max = q_interval.sup - Rational(1,1000000) if q_interval.right_open else q_interval.sup
    q_example = 0.5 if q_interval.contains(0.5) else grid_q
    y = [ VMInterval("Verkaufspreis"), VMInterval("Verkaufsmenge/Anbieter") ]
    for i, x in enumerate([ v_sne, m_sne ]):
        if x.has(q):
            # Verkaufspreis resp. -menge für q = (q_min, q_example, q_max)
            y_values = [ simplify(x.subs(q, q_min)), simplify(x.subs(q, q_example)), simplify(x.subs(q, q_max)) ]
            y[i].interval = Interval(min(y_values), max(y_values))
            y[i].example = y_values[1]
            # x(q) kann Extrema haben. Für die min und max-Werte von x also dx/dq = 0 mit Randbedingung q innerhalb q_interval berechnen
            dx_dq = diff(x, q)
            q_0_sol = solve(Eq(dx_dq, 0), q)
            for q_0 in q_0_sol:
                if q_interval.contains(q_0):
                    y_0 = simplify(x.subs(q, q_0))
                    y[i].interval = Interval(min(y[i].interval.inf, y_0), max(y[i].interval.sup, y_0))
        else:
            y[i].interval = Interval(x, x)
            y[i].example = x
    return y[0], y[1]

# Eine Gleichgewichtsbedingung ist erfüllbar, wenn eine gegebene se_condition erfüllbar ist.
def equilibrium_satisfiable(title, se_condition, se_symbolic, q, q_condition, logger, grid_step = 0.01):
    
    # Randbedingung für Wahrscheinlichkeitswerte
    if se_condition.has(q):
        se_condition = And(se_condition, q_condition)
    
    # Erfüllbarkeit der Gleichgewichtsbedingung
    if se_condition == S.true:
        logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht immer erfüllt")
        q_interval = intersect_q_intervals(q_condition, q, logger)
        return SatisfiableResult(True, (q_interval.inf + q_interval.sup) / 2, q_interval)
    if se_condition == S.false:
        logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewichtsbedingung nicht erfüllbar")
        return SatisfiableResult()
    if not se_symbolic:
        se_rels = list(se_condition.args) if isinstance(se_condition, And) else [se_condition]
        if se_condition.has(q):
            se_solved = reduce_inequalities(se_rels, q)
        else:
            se_solved = reduce_inequalities(se_rels)
    else:
        se_solved = satisfiable(se_condition)
    if se_solved == S.false:
        logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Kein Gleichgewicht, Gleichgewichtsbedingung nicht erfüllbar")
        return SatisfiableResult()
    
    # Numerische Lösung für nicht-symbolische Ungleichungen suchen
    if not se_symbolic:
        # Allgemeine Lösung: Liste von Intervallen, die se_condition erfüllen
        q_interval = Interval(q, q) if q == 0 or q == 1 else Interval.open(0, 1)
        se_sol_intervals = search_for_q_intervals(se_solved, q, q_interval, logger) 
        if len(se_sol_intervals) > 0:
            q_example = (se_sol_intervals[0].inf + se_sol_intervals[0].sup) / 2
            se_sol_intervals = se_sol_intervals[0] if len(se_sol_intervals) == 1 else se_sol_intervals
            logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht gdw. {q} in {se_sol_intervals}, z.B. {q} = {q_example}")
            return SatisfiableResult(True, q_example, se_sol_intervals)
        logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewichtsbedingung nicht erfüllbar")
        return SatisfiableResult()
        # Beispielswerte für Lösungen aus grid
        # grid_satisfiable, grid_q = probability_grid_check(se_condition, q, grid_step)
        # if not grid_satisfiable == False:
        #     logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht gdw. {se_solved}, z.B. {q} = {grid_q}")
        #     return SatisfiableResult(grid_satisfiable, nsimplify(grid_q, rational=True), intersect_q_intervals(se_solved, q, logger))
        # logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht gdw. {se_solved}, aber keine grid-Lösung für {q} gefunden")
        # return SatisfiableResult(se_solved)
    
    logger.print(HilbertLogger.Verbosity.INFO, f"{title}: Gleichgewicht gdw. {se_solved}")
    return SatisfiableResult(se_solved)

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
def nash_equilibrium(N, MNR, cfg, logger, functionname="nash_equilibrium"): # N = Anzahl Tankstellen, MNR = M/N, M = Anzahl Kunden

    # Preise: v1 = "Abweichler" (Tankstelle 1), vo = "alle anderen Tankstellen"
    v1, vo, v = symbols("v1 vo v", real=True)

    # Durchschnittspreis bei (v1, vo,...,vo)
    v_mean = (v1 + (N - 1) * vo) / N

    # Die Gesamtnachfrage (gekaufte Menge) hängt vom Durchschnittspreis ab
    m_sum = MNR * N * (cfg.m_0 - cfg.h * (v_mean - cfg.v_0))

    # An der Tankstelle verkaufte Menge: m[n] ∝ (m_sum/N) + r * (v_mean - v[n]) mit Proportionalitätsfaktor C_mv = 1 aus C_mv * sum((m_sum/N) + r * (v_mean - v[n])) = m_sum
    m1  = (m_sum / N) + cfg.r * (v_mean - v1) # Verkaufsmenge der Tankstelle 1

    # Nutzen von Tankstelle 1
    U1 = m1 * (v1 / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.F_n - cfg.K_n

    # Nutzen U1(v1, vo) der Tankstelle 1 maximieren (First-Order-Condition, FOC: ∂U1/∂v1 = 0)
    dU1_dv1 = diff(U1, v1)
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {MNR * N}, N = {N}): U1 = {U1}")
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {MNR * N}, N = {N}): dU1/dv1 = {dU1_dv1}")

    # Symmetrisches Gleichgewicht: v1 = vo = v
    foc_sym = simplify(dU1_dv1.subs({v1: v, vo: v}))

    # Löse FOC im symmetrischen Punkt nach v, ∂U/∂v = 0
    v_sne = solve(Eq(foc_sym, 0), v, dict=True)
    assert_or_die(len(v_sne) > 0, f"len({v_sne}) > 0", logger)
    v_sne = factor(simplify(v_sne[0][v]))

    # Verkaufsmenge m_sne und Nutzen U_sne pro Tankstelle im Gleichgewicht
    m_sne = MNR * (cfg.m_0 - cfg.h * (v_sne - cfg.v_0))
    U_sne = m_sne * (v_sne / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.F_n - cfg.K_n
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {MNR * N}, N = {N}): v_sne = {v_sne}, m_sne = {m_sne}, U_sne = {U_sne}")

    # Aufgrund des linearen Zusammenhangs zwischen Verkaufsmenge und Verkaufspreis kann die verkaufte Menge an einer besonders teuren Tankstelle rechnerisch auch negativ werden.
    # Check gegen negative Mengen:
    if m_sne.is_Number and m_sne < 0:
        logger.print(HilbertLogger.Verbosity.WARNING, f"{functionname}(M = {MNR * N}, N = {N}): m_sne = {m_sne}, non-negative value expected")
    m_sum = simplify(m_sum.subs({v1: v_sne, vo: v_sne}))
    return v_sne, m_sne, U_sne, m_sum

# Plot Gleichgewichte (aktive Gruppen, Verkaufspreise) für gebebene Werte von N (z.B. oo), q (z.B. 0.5) und Strategie ("T1f" oder "T1af") in Abhängigkeit von L/N und M/(N+L)
def plot_json_results(json_filename = "hilbert_gas_stations_inf_results.json", png_filename = "hilbert_gas_stations_inf_results.png", plot_N = oo, plot_strategies = ["T1f", "T1af"], plot_q = 0.5, logger = HilbertLogger(verbosity = HilbertLogger.Verbosity.INFO)):
    if Path(json_filename).is_file():
        with open(json_filename, "r") as json_file:
            json_examples = json.load(json_file)
            fig = plt.figure(figsize=(10, 10))
            for plot_n, plot_strategy in enumerate(plot_strategies):
                q = symbols("q", real = True)
                plot_LNR = []
                plot_MNLR = []
                plot_num_active_groups = []
                plot_v = []
                # Parsing json file, Deserialisierung und Konvertierung
                for json_example in json_examples:
                    json_N, json_LNR, json_MNLR, json_strategies = json_example
                    if json_N != srepr(plot_N):
                        continue
                    for json_strategy, json_equilibriums in json_strategies.items():
                        if json_strategy != plot_strategy:
                            continue
                        for json_equilibrium in json_equilibriums:
                            equilibrium = HilbertGroups.Equilibrium(False, json_strategy, {}, False, 0, 0)
                            equilibrium.strategy = json_equilibrium[0]
                            equilibrium.status = json_equilibrium[1]
                            for group_name, json_group in json_equilibrium[2].items():
                                equilibrium.groups[group_name] = HilbertGroup(group_name, None, None, None, logger = logger)
                                equilibrium.groups[group_name].name = json_group[0]
                                equilibrium.groups[group_name].N = sympify(json_group[1])
                                equilibrium.groups[group_name].v_sne = sympify(json_group[2])
                                equilibrium.groups[group_name].m_sne = sympify(json_group[3])
                                equilibrium.groups[group_name].U["f"] = sympify(json_group[4])
                                equilibrium.groups[group_name].U["a"] = sympify(json_group[5])
                            equilibrium.q_condition = sympify(json_equilibrium[3])
                            equilibrium.v = VMInterval(name = "Verkaufspreis", example = sympify(json_equilibrium[4]), interval = sympify(json_equilibrium[5]))
                            equilibrium.m = VMInterval(name = "Verkaufsmenge/Anbieter", example = sympify(json_equilibrium[6]), interval = sympify(json_equilibrium[7]))
                            if reduce_inequalities(equilibrium.q_condition.subs(q, plot_q)) == S.false:
                                continue
                            if isinstance(equilibrium.q_condition,FiniteSet) and equilibrium.q_condition.contains(plot_q) == S.false:
                                continue
                            plot_LNR.append(float(sympify(json_LNR)))
                            plot_MNLR.append(float(sympify(json_MNLR)))
                            plot_num_active_groups.append(len(equilibrium.groups))
                            plot_v.append(float(equilibrium.v.example))
                            logger.print(HilbertLogger.Verbosity.DEBUG, f"JSON: N={json_N}, L/N={json_LNR}, M/(N+L)={json_MNLR}, strategy={json_strategy}, equilibrium={equilibrium}")
                # Plot Anzahl aktiver Gruppen und Verkaufspreise über L/N und M/(N+L) für gebebene N, q, Strategie
                ax = fig.add_subplot(2, 1, plot_n + 1, projection='3d')
                plot_LNR_dict = {}
                for i, lnr in enumerate(plot_LNR):
                    if lnr not in plot_LNR_dict:
                        plot_LNR_dict[lnr] = [[], [], []]
                    plot_LNR_dict[lnr][0].append(plot_LNR[i])
                    plot_LNR_dict[lnr][1].append(plot_MNLR[i])
                    plot_LNR_dict[lnr][2].append(plot_v[i])
                for lnr, values in plot_LNR_dict.items():
                    ax.plot3D(values[0], values[1], values[2])
                ax.scatter3D(plot_LNR, plot_MNLR, plot_v, c=plot_num_active_groups)
                ax.set_title(f"Verkaufspreis v über L/N und M/(N+L) für N={plot_N}, q={plot_q}, Strategie={plot_strategy}, Beispielwerte:")
                ax.set_xlabel("L/N")
                ax.set_ylabel("M/(N+L)")
                ax.set_zlabel("v")
            plt.savefig(png_filename)
            plt.show()

def equilibrium_N_player():

    # Tankstellen mit M Kunden, N etablierte Tankstellen und L neuen Konkurrenten:
    # Setup mit 4 Gruppen: T1R (reiche etablierte Tankstellen), T1A (arme etablierte Tankstellen), T2R (reiche neue Konkurrenten), T2A (arme neue Konkurrenten)
    # Gruppen mit Nutzen U_g < 0 werden entfernt und die Gleichgewichte ermittelt.
    logger = HilbertLogger(verbosity = HilbertLogger.Verbosity.INFO, filename = "hilbert_gas_stations_equilibrium.log")
    logger.print(HilbertLogger.Verbosity.INFO, f"Berechnung der Gleichgewichte für Tankstellen mit M Kunden, N etablierten Tankstellen und L neuen Konkurrenten (Konkurrentenspiel mit Markteintritt):\n")

    # Symbolisch: N etablierte Tankstellen, LNR = L/N, L neue Konkurrenten, MNLR = M/(N+L), M Kunden
    groups = HilbertGroups(symbolic = True, logger = logger) 
    groups.group_equilibrium(N = symbols("N", int=True, positive=True), LNR = symbols("LNR", real=True), MNLR = symbols("MNLR", real=True))
    
    # Beispielwerte: N etablierte Tankstellen, LNR = L/N, L = LNR*N neue Konkurrenten, MNLR = M/(N+L), M = MNLR*(N+L) = MNLR*(N+LNR*N) Kunden
    groups = HilbertGroups(symbolic = False, logger = logger)
    hilbert_NLM_examples, hilbert_inf_examples = [], [] # Beispiele für endliche N und N -> oo
    hilbert_NLM_results, hilbert_inf_results = {}, {}   # Gleichgewichte der Beispiele für endliche N und N -> oo

    # Beispiele für Hilbert-Tankstellen mit endlichen N, LNR=L/N, MNLR=M/(N+L)
    hilbert_NLM_input = [
        (   1, 1, Rational(1,2)),    #    1 etablierte Tankstelle,     1 neuer Konkurrent,     1 Kunde
        (   1, 1, Rational(2,2)),    #    1 etablierte Tankstelle,     1 neuer Konkurrent,     2 Kunden
        (   1, 1, Rational(3,2)),    #    1 etablierte Tankstelle,     1 neuer Konkurrent,     3 Kunden
        (   2, 1, Rational(4,4)),    #    2 etablierte Tankstellen,    2 neue Konkurrenten,    4 Kunden
        (   2, 1, Rational(6,4)),    #    2 etablierte Tankstellen,    2 neue Konkurrenten,    6 Kunden
        (1000, 1, Rational(1,2)),    # 1000 etablierte Tankstellen, 1000 neue Konkurrenten, 1000 Kunden
        (1000, 1, Rational(3,4)),    # 1000 etablierte Tankstellen, 1000 neue Konkurrenten, 1500 Kunden
        (1000, 1, 1),                # 1000 etablierte Tankstellen, 1000 neue Konkurrenten, 2000 Kunden
        (1000, 1, 2),                # 1000 etablierte Tankstellen, 1000 neue Konkurrenten, 4000 Kunden
        (1000, Rational(1,1000), 2), # 1000 etablierte Tankstellen,    1 neuer Konkurrent,  2002 Kunden
        (   1, 1000, 2),             #    1 etablierte Tankstellen, 1000 neue Konkurrenten, 2002 Kunden
    ]
    hilbert_NLM_examples = []
    if True:
        for nlm in hilbert_NLM_input:
            result = groups.group_equilibrium(N = nlm[0], LNR = nlm[1], MNLR = nlm[2])
            hilbert_NLM_examples.append((nlm, result))
    
    # Beispiele für Hilbert-Tankstellen mit Grenzübergang N -> oo vielen etablierten Tankstellen, L -> oo vielen neuen Konkurrenten, M -> oo vielen Kunden und endlichen Dichten L/N und M/(N+L)
    hilbert_inf_examples = []
    # for nlm in [ (oo, 1, 1/4), (oo, 1, 1), (oo, 1, 100), (oo, 100, 1), (oo, 100, 2), (oo, 100, 100) ]:
    #     result = groups.group_equilibrium(N = nlm[0], LNR = nlm[1], MNLR = nlm[2])
    #     hilbert_inf_examples.append((nlm, result))
    # for example in hilbert_inf_examples:
    #     nlm, results = example[0], example[1]
    #     for strategy, result in results.items():
    #         for r in result:
    #             logger.print(HilbertLogger.Verbosity.INFO, f"N={nlm[0]}, L/N={nlm[1]}, L={nlm[1] * nlm[0]}, M/(N+L)={nlm[2]}, M={nlm[2] * (nlm[0] + nlm[1] * nlm[0])}, {r}")
    #     logger.print(HilbertLogger.Verbosity.INFO, f"")
    hilbert_inf_input = []
    for MNLR in [ Rational(1,1000), Rational(1,4), Rational(1,2) ]: # N -> oo, L -> oo, M -> oo, M/(N+L) = [ 1/1000, 0.25, 0.5 ]
        for LNR in [ Rational(1,1000), Rational(1,4), Rational(1,2) ]: # N -> oo, L -> oo, L/N = [ 1/1000, 0.25, 0.5 ]
            hilbert_inf_input.append(( oo, LNR, MNLR))
    for MNLR in [ 1, 2, 10, 25, 50, 75, 100 ]: # N -> oo, L -> oo, M -> oo, M/(N+L) = [ 1, 2, 10, 25, 50, 75, 100 ]
        for LNR in [ 1, 2, 10, 25, 50, 75, 100 ]: # N -> oo, L -> oo, L/N = [ 1, 2, 10, 25, 50, 75, 100 ]
            hilbert_inf_input.append(( oo, LNR, MNLR))
    if True:
        hilbert_inf_json = []
        for nlm in hilbert_inf_input:
            result = groups.group_equilibrium(N = nlm[0], LNR = nlm[1], MNLR = nlm[2])
            hilbert_inf_examples.append((nlm, result))
            hilbert_inf_json.append(tojson((nlm[0], nlm[1], nlm[2], result)))
        with open(f"hilbert_gas_stations_inf_results.json", "w") as json_file:
            json.dump(hilbert_inf_json, json_file, indent=4)

    if len(hilbert_NLM_examples + hilbert_inf_examples) > 0:
        logger.print(HilbertLogger.Verbosity.INFO, f"Beispiele für N etablierte Tankstellen, L neue Konkurrenten, M Kunden:\n")
        for example in hilbert_NLM_examples + hilbert_inf_examples:
            nlm = example[0]
            results = example[1]
            for strategy, result in results.items():
                for r in result:
                    N = nlm[0]
                    L = nlm[1] * N
                    M = nlm[2] * (N + L)
                    logger.print(HilbertLogger.Verbosity.INFO, f"N={N}, L/N={nlm[1]}, L={L}, M/(N+L)={nlm[2]}, M={M}, {r}")
            logger.print(HilbertLogger.Verbosity.INFO, f"")

    # Plot Gleichgewichte (aktive Gruppen, Verkaufspreise) für N -> oo, q = 0.5 und unterschiedliche Strategien in Abhängigkeit von L/N und M/(N+L)
    plot_json_results(json_filename = "hilbert_gas_stations_inf_results.json", png_filename = "hilbert_gas_stations_inf_results.png", plot_N = oo, plot_strategies = ["T1f", "T1af"], plot_q = 0.5, logger = logger)

if __name__ == "__main__":
    equilibrium_N_player()
