"""
Nash-Gleichgewicht in Hilbert-Tankstellen:
Beschreibung und Erlaeuterungen in ../docs/HilbertTankstellen.md
"""
import math
from enum import IntEnum
from sympy import *
from spb import * # pip install sympy_plot_backends

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
        self.f_n = symbols("f_n", real=True) if symbolic else    0  if is_established else 0.5 # laufende Abschreibung im Falle von Investitionskosten, z.B. 0 (etablierte Tankstelle) oder 0.5 Euro/Min (Neueröffnung einer Tankstelle)
        self.K_n = symbols("K_n", real=True) if symbolic else    0  if is_rich else 1 # Kosten eines Konkurrenzkampf, z.B. 0 (reiche Tankstelle mit Rücklagen) oder 1 Euro/Min (arme Tankstelle ohne Rücklagen)
        self.s   = symbols("s",   real=True) if symbolic else 0.19  # Umsatzsteuer, z.B. 0.19 (19% Mwst.)
        # Arbeitspunkte: Hebel, Basismenge und Basismengenpreis sind frei konfigurierbar; die verkaufte Gesamtmenge sum(m[n]) reduziert sich linear mit dem durchschnittlichen Verkaufspreis v_mean
        self.m_0 = symbols("m_0", real=True) if symbolic else   10  # Basismenge, z.B. 10 liter
        self.v_0 = symbols("v_0", real=True) if symbolic else    2  # Basismengenpreis, z.B. 2 Euro/liter
        self.h   = symbols("h",   real=True) if symbolic else (1/2)*(self.m_0/self.v_0)  # Hebel in liter/Euro: m = M * (m_0 - h * (v - v_0)), z.B. h = (1/2)*(m_0/v_0) halbiert die verkaufte Menge bei doppeltem Preis
        self.r   = symbols("r",   real=True) if symbolic else  100  # Rate r modelliert wie sensibel die Kunden auf lokale Preisunterschiede reagieren, Verkaufsmenge m[n] proportional zu (m_sum/N) + r * (v_mean - v[n])
        # Nur für Tankstellen-Modell mit Servicegebühren:
        self.lambd = symbols("lambda", real=True) if symbolic else 100 # Bei höheren Gebühren g wechseln Kunden zur Konkurrenz: Anzahl Kunden = M_n = M_active/N - cfg.lambd * (g_mean - g_n)
        self.eta   = symbols("eta",    real=True) if symbolic else   0 # Bei höheren Durchschnittsgebühren g_mean schrumpft die Gesamtzahl der Kunden: M_active = M - cfg.eta * N * g_mean

# Für alle Argumente in *values: x = 0 if abs(x) < eps else x
def epsilon2zero(*values, eps = 1.0e-6):
    result = tuple(0 if abs(float(x)) < eps else x for x in values)
    return result[0] if len(result) == 1 else result

# Für alle Argumente in *values: x = x.ljust(max(*values))
def ljustify(*values):
    arg_lengths = tuple(len(s) for s in values)
    max_len = max(arg_lengths)
    result = tuple(s.ljust(max_len) for s in values)
    return result[0] if len(result) == 1 else result

# Nash-Gleichgewicht bei M Kunden und N Tankstellen ohne Markteintritte.
# Klassische, lehrbuchmäßige Berechnung über N variable Preise; langsam und nur für kleine N zum Vergleich.
def nash_equilibrium_classical(M, N, cfg, logger, functionname="nash_equilibrium_classical"):
    
    # Die Gesamtmenge m an verkauftem Benzin sei umgekehrt proportional zum Durchschnittspreis v
    v = [ symbols(f"v[{n}]", real=True) for n in range(1, N+1) ] # Tankstellen-Verkaufspreise, z.B. 2 Euro/liter
    v_mean = (1/N) * sum(v) # Durchschnittspreis
    m_sum_buyer = M * (cfg.m_0 - cfg.h * (v_mean - cfg.v_0)) # Gekaufte Gesamtmenge in Abhängigkeit vom Durchschnittspreis v_mean

    # Die Verkaufsmenge m[n] an jeder Tankstelle sei linear zur Preisdifferenz v_mean - v[n]
    m_proportional_to = [ ((m_sum_buyer/N) + cfg.r * (v_mean - v[n])) for n in range(N) ] # An der Tankstelle verkaufte Menge: m[n] ∝ (m_sum/N) + r * (v_mean - v[n])
    C_mv = m_sum_buyer / sum(m_proportional_to) # m[n] = C_mv * m_proportional_to mit Proportionalitätsfaktor C_mv aus m_sum_seller = C_mv * sum(m_proportional_to) = m_sum_buyer
    m_sum_seller = C_mv * sum(m_proportional_to) # Verkaufte Gesamtmenge
    assert(simplify(m_sum_seller - m_sum_buyer) == 0) # Verkaufte Gesamtmenge = Gekaufte Gesamtmenge
    
    # Schnelle Berechnung des Verkaufspreises: Wir suchen ein symmetrisches Nash-Gleichgewicht (SNE). Im SNE verkaufen alle Tankstellen zum gleichen Preis: 
    # v[n] = v[0] für alle 0<n<N. SNE vorausgesetzt, setzen wir v[n] = v[0] für alle 0<n<N und berechnen nur noch einen Preis, nämlich v[0].
    U0 = C_mv * m_proportional_to[0] * (v[0] / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.f_n - cfg.K_n # Nutzenfunktion der ersten Tankstelle
    dUdV0 = diff(U0, v[0]) # dUdV = ∂U[0]/∂v[0]
    for n in range(1,N):
        dUdV0 = dUdV0.subs(v[n], v[0])
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): dU0/dv0 = {simplify(dUdV0)} (assuming nash equilibrium exists)")
    v0_v_sne = solve(dUdV0, v[0]) # Gewinnmaximierung: ∂U[0]/∂v[0] = 0
    if len(v0_v_sne) > 0:
        v0_v_sne = v0_v_sne[0]
    v_sne = {}
    for n in range(N):
        v_sne[v[n]] = v0_v_sne
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): v0_v_sne = {v0_v_sne} (assuming nash equilibrium exists)")
    
    # Ist N klein genug, können wir alle Nutzenfunktionen U[n] berechnen, ∂U[n] / ∂v[n] = 0 für alle N lösen und das Ergebnis überprüfen
    if N <= 10:
        # Nutzenfunktionen der Tankstellen
        m = [ C_mv * m_proportional_to[n] for n in range(N) ] # m[n] = an der n-ten Tankstelle verkaufte Menge
        U = [ m[n] * (v[n] / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.f_n - cfg.K_n for n in range(N) ] # U[n] = Nutzenfunktion der n-ten Tankstelle
        logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): U = {U}")

        # Im Nash-Gleichgewicht kann keine Tankstelle ihren Gewinn durch einseitige Strategieänderung steigern, 
        # d.h. im Nash-Gleichgewicht gilt für alle Tankstellen: ∂U[n] / ∂v[n] = 0
        dUdV = [ diff(U[n], v[n]) for n in range(N) ] # dUdV = [ ∂U[1]/∂v[1], ∂U[2]/∂v[2], ... ∂U[N]/∂v[N] ]
        for n in range(N):
            logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): dU/dv{n+1} = {simplify(dUdV[n])}")
        v_sne = solve(dUdV, v) # i.e. ∂U[n] / ∂v[n] = 0 for all n (time consuming with symbolic parameters!)
        logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): v_sne = {v_sne}")
        for n in range(N):
            if not math.isclose(v_sne[v[n]], v0_v_sne, rel_tol=1e-6, abs_tol=1e-6):
                logger.print(HilbertLogger.Verbosity.WARNING, f"{functionname}(M = {M}, N = {N}): v_sne[{v[n]}] = {v_sne[v[n]]}, expected value v0_v_sne = {v0_v_sne}")
    
    # Verkaufsmenge m_sne und Nutzen U_sne pro Tankstelle im Gleichgewicht
    m_sne = M * (cfg.m_0 - cfg.h * (v0_v_sne - cfg.v_0)) / N
    U_sne = m_sne * (v0_v_sne / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.f_n - cfg.K_n

    # Aufgrund des linearen Zusammenhangs zwischen Verkaufsmenge und Verkaufspreis kann die verkaufte Menge an einer besonders teuren Tankstelle rechnerisch auch negativ werden.
    # Check gegen negative Mengen:
    if m_sne < 0:
        logger.print(HilbertLogger.Verbosity.WARNING, f"{functionname}(M = {M}, N = {N}): m_sne = {m_sne}, non-negative value expected")
    return v0_v_sne, m_sne, U_sne

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
def nash_equilibrium_one_deviant(M, N, cfg, logger, functionname="nash_equilibrium_one_deviant"):

    # Preise: v1 = "Abweichler" (Tankstelle 1), vo = "alle anderen Tankstellen"
    v1, vo, v = symbols("v1 vo v", real=True)

    # Durchschnittspreis bei (v1, vo,...,vo)
    v_mean = (v1 + (N - 1) * vo) / N

    # Die Gesamtnachfrage (gekaufte Menge) hängt vom Durchschnittspreis ab
    m_sum = M * (cfg.m_0 - cfg.h * (v_mean - cfg.v_0))

    # An der Tankstelle verkaufte Menge: m[n] ∝ (m_sum/N) + r * (v_mean - v[n]) mit Proportionalitätsfaktor C_mv = 1 aus C_mv * sum((m_sum/N) + r * (v_mean - v[n])) = m_sum
    m1  = (m_sum / N) + cfg.r * (v_mean - v1) # Verkaufsmenge der Tankstelle 1

    # Nutzen von Tankstelle 1
    U1 = m1 * (v1 / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.f_n - cfg.K_n

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
    U_sne = m_sne * (v_sne / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.f_n - cfg.K_n
    logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}): v_sne = {v_sne}, m_sne = {m_sne}, U_sne = {U_sne}")

    # Aufgrund des linearen Zusammenhangs zwischen Verkaufsmenge und Verkaufspreis kann die verkaufte Menge an einer besonders teuren Tankstelle rechnerisch auch negativ werden.
    # Check gegen negative Mengen:
    if m_sne.is_Number and m_sne < 0:
        logger.print(HilbertLogger.Verbosity.WARNING, f"{functionname}(M = {M}, N = {N}): m_sne = {m_sne}, non-negative value expected")
    return v_sne, m_sne, U_sne

# Symmetrisches Nash-Gleichgewicht (Literpreis, Gebühr) via "ein Abweichler"-Ansatz
def nash_equilibrium_price_and_fee(M, N, cfg, logger, functionname="nash_equilibrium_price_and_fee"):

    try:
        # Literpreise: v1 = "Abweichler" (Tankstelle 1), vo = "alle anderen Tankstellen"
        v1, vo, v = symbols("v1 vo v", real=True)

        # Durchschnittlicher Literpreis bei (v1, vo,...,vo)
        v_mean = (v1 + (N - 1) * vo) / N

        # Gebühren: G1 = "Abweichler" (Tankstelle 1), Go = "alle anderen Tankstellen"
        g1, go, g = symbols("g1 go g", real=True)

        # Durchschnittliche Gebühren bei (v1, vo,...,vo)
        g_mean = (g1 + (N - 1) * go) / N

        # q = m_sum / M_active = Menge pro Kunde
        q = symbols("q", real=True) # q = m_sum / M_active später einsetzen

        # v_eff = effektiver Literpreis inkl. Servicegebühr
        v_eff_o = vo + go / q
        v_eff_1 = v1 + g1 / q
        v_eff_mean = (v_eff_1 + (N - 1) * v_eff_o) / N # v_eff_mean = effektiver Durchschnittspreis

        # M_active = Gesamtanzahl handelnder Kunden abhängig von der Durchschnittsgebühr, M = potentielle, maximale Kundenzahl (M_active = M wenn alle Kunden kaufen), N = Anzahl Anbieter
        # M_active = M - cfg.eta * N * g_mean # Negative Gebühren können rechnerisch attraktiv sein, weil durch M_active > M die Anzahl handelnder Kunden größer wird als erlaubt
        M_active = Piecewise((M - cfg.eta * N * g_mean, M - cfg.eta * N * g_mean <= M), (0, M - cfg.eta * N * g_mean < 0), (M, True))

        # M_1 = Anzahl Kunden der 1-ten Tankstelle, Mo = Anzahl Kunden "alle anderen Tankstellen"
        Mo = M_active/N + cfg.lambd * (g_mean - go)
        M1 = M_active/N + cfg.lambd * (g_mean - g1)

        # m_sum = Gesamtnachfrage (gekaufte Menge) abhängig vom effektiven Durchschnittspreis
        m_sum = M_active * (cfg.m_0 - cfg.h * (v_eff_mean - cfg.v_0))

        # m1 = verkaufte Menge der 1. Tankstelle abhängig von Preisdifferenz zur Konkurrenz (Literpreis und Gebühr)
        # m1  = (m_sum / N) + cfg.r * (v_mean - v1) # Kauf abhängig vom Literpreis führt zu Gleichgewicht mit Gebühren
        m1  = (m_sum / N) + cfg.r * (v_eff_mean - v_eff_1) # Kauf abhängig vom effektiven Verkaufspreis führt zu Gleichgewicht ohne Gebühren

        # U1 = Nutzenfunktion der 1. Tankstelle
        U1 = m1 * (v1 / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.f_n - cfg.K_n + M1 * g1 / (1 + cfg.s)

        # Nutzen U1(v1, vo) und U1(g1, go) der Tankstelle 1 maximieren (First-Order-Condition, FOC: ∂U1/∂v1 = 0, ∂U1/∂g1 = 0), noch abhängig von q (Menge pro Kunde) als freiem Symbol
        # q = cfg.m_0 - cfg.h*((v + g/q) - cfg.v_0)) <=> F(q,v,g) = q**2 - (cfg.m_0 - cfg.h*(v - cfg.v_0))*q + cfg.h*g = 0
        # Implizite Differentiation: 
        # F(x,f(x)) = 0 <=> ∂F/∂x + ∂F/∂f(x) * df/dx = 0 <=> df/dx = -(∂F/∂x) / (∂F/∂f(x))
        # F(x,y) = 0 <=> (∂F/∂x)dx + (∂F/∂y)dy = 0 <=> dx/dy = -(∂F/∂y) / (∂F/∂x)
        # siehe https://de.wikipedia.org/wiki/Totales_Differential, https://de.wikipedia.org/wiki/Implizite_Differentiation
        F = q**2 - (cfg.m_0 - cfg.h*(v_mean - cfg.v_0))*q + cfg.h*g_mean
        dq_dv1 = -diff(F, v1)/diff(F, q)
        dq_dg1 = -diff(F, g1)/diff(F, q)
        dU_dv1_total = diff(U1, v1) + diff(U1, q)*dq_dv1
        dU_dg1_total = diff(U1, g1) + diff(U1, q)*dq_dg1

        # Symmetrisches Gleichgewicht: v1 = vo = v, g1 = go = g
        sym_subs = {v1: v, vo: v, g1: g, go: g}
        foc_sym_v = simplify(dU_dv1_total.subs(sym_subs))
        foc_sym_g = simplify(dU_dg1_total.subs(sym_subs))
        F = simplify(F.subs(sym_subs))

        # Numerische Lösung ∂U/∂v = 0, ∂U/∂g = 0, F = 0 via nsolve() mit Startwerten, solve() liefert nur leere List
        v0_guess = cfg.e_n
        g0_guess = 0
        q0_guess = float(cfg.m_0 - cfg.h*(v0_guess - cfg.v_0))
        sol = nsolve([Eq(foc_sym_v, 0), Eq(foc_sym_g, 0), Eq(F, 0)], [v, g, q], [v0_guess, g0_guess, q0_guess])

        # Literpreis v_sne, effektiver Preis v_eff_sne, Anzahl handelnder Kunden M_active_sne, Menge m_sne und Nutzen U_sne pro Tankstelle im Gleichgewicht
        v_sne, g_sne, q_sne = map(float, sol)
        v_eff_sne = v_sne + g_sne / q_sne
        M_active_sne = M - cfg.eta * N * g_sne
        m_sne = M_active_sne * (cfg.m_0 - cfg.h * (v_eff_sne - cfg.v_0)) / N
        U_sne = m_sne * (v_sne / (1 + cfg.s) - cfg.e_n) - cfg.b_n - cfg.f_n - cfg.K_n + (M_active_sne/N) * g_sne / (1 + cfg.s)
        logger.print(HilbertLogger.Verbosity.DEBUG, f"{functionname}(M = {M}, N = {N}, lambda = {cfg.lambd}, eta = {cfg.eta}): v_sne = {v_sne:.4f}, v_eff_sne = {v_eff_sne:.4f}, g_sne = {g_sne:.4f}, m_sne = {m_sne:.4f}, U_sne = {U_sne:.4f}, M_active_sne = {M_active_sne:.4f}, q_sne = {q_sne:.4f}")

        # Check gegen Randbedingungen:
        v_sne, v_eff_sne, g_sne, m_sne, U_sne, M_active_sne, q_sne = epsilon2zero(v_sne, v_eff_sne, g_sne, m_sne, U_sne, M_active_sne, q_sne)
        if m_sne < 0 or q_sne < 0 or M_active_sne < 0:
            logger.print(HilbertLogger.Verbosity.WARNING, f"{functionname}(M = {M}, N = {N}, lambda = {cfg.lambd}, eta = {cfg.eta}): m_sne = {m_sne}, q_sne = {q_sne}, M_active_sne = {M_active_sne}, non-negative value(s) expected")
        if M_active_sne > M:
            logger.print(HilbertLogger.Verbosity.WARNING, f"{functionname}(M = {M}, N = {N}, lambda = {cfg.lambd}, eta = {cfg.eta}): M_active_sne = {M_active_sne} must not be greater than M = {M}")

        return v_sne, m_sne, U_sne, v_eff_sne, g_sne
    
    except Exception as exc:
        logger.print(HilbertLogger.Verbosity.WARNING, f"{functionname}(M = {M}, N = {N}, lambda = {cfg.lambd}, eta = {cfg.eta}): exception {exc}")
    return 0, 0, 0, 0, 0

def nash_equilibrium_hilbert_with_fees():

    # Nash-Gleichgewichte bei Hilbert-Tankstellen mit Literpreisen und extra Servicegebühren pro Einkauf
    logger = HilbertLogger(verbosity = HilbertLogger.Verbosity.INFO)
    cfg = HilbertParameter(symbolic = False)
    logger.print(HilbertLogger.Verbosity.INFO, f"Nash-Gleichgewicht (Symmetrical Nash Equilibrium, SNE) bei Hilbert-Tankstellen mit Literpreis und Gebühren:\n")
    for lambda_eta in [ (100, 0), (0.1, 0), (100, 0.1) ]:
        cfg.lambd = lambda_eta[0] # Bei höheren Gebühren g wechseln Kunden zur Konkurrenz: Anzahl Kunden = M_n = M_active/N - cfg.lambd * (g_mean - g_n)
        cfg.eta = lambda_eta[1]   # Bei höheren Durchschnittsgebühren g_mean schrumpft die Gesamtzahl der Kunden: M_active = M - cfg.eta * N * g_mean
        for MN in [ (2,2), (3,3), (10,10), (1,1000000), (1000000,1000000), (100000000,1000000) ]:
            v_sne, m_sne, U_sne = nash_equilibrium_one_deviant(M = MN[0], N = MN[1], cfg = cfg, logger = logger) # Zum Vergleich: SNE ohne Servicegebühren
            desc1, desc2 = ljustify(f"SNE(M = {MN[0]}, N = {MN[1]}, keine Gebühren erlaubt):", f"SNE(M = {MN[0]}, N = {MN[1]}, lambda = {cfg.lambd}, eta = {cfg.eta}):")
            logger.print(HilbertLogger.Verbosity.INFO, f"{desc1} v_sne = {v_sne:.4f}, m_sne = {m_sne:.4f}, U_sne = {U_sne:.4f}")
            v_sne, m_sne, U_sne, v_eff_sne, g_sne = nash_equilibrium_price_and_fee(M = MN[0], N = MN[1], cfg = cfg, logger = logger) # SNE mit Servicegebühren
            logger.print(HilbertLogger.Verbosity.INFO, f"{desc2} v_sne = {v_sne:.4f}, m_sne = {m_sne:.4f}, U_sne = {U_sne:.4f}, v_eff_sne = {v_eff_sne:.4f}, g_sne = {g_sne:.4f}, Gebühren erlaubt")
        logger.print(HilbertLogger.Verbosity.INFO, f"")

def nash_equilibrium_hilbert():
    
     # Nash-Gleichgewichte bei Hilbert-Tankstellen
    logger = HilbertLogger(verbosity = HilbertLogger.Verbosity.INFO)
    logger.print(HilbertLogger.Verbosity.INFO, f"Nash-Gleichgewicht (Symmetrical Nash Equilibrium, SNE) bei Hilbert-Tankstellen (Literpreis, keine Gebühren):\n")

    # Nash-Gleichgewicht mit symbolischer Lösung (M = Anzahl Kunden, N = Anzahl Anbieter)
    v_sne, m_sne, U_sne = nash_equilibrium_one_deviant(M = symbols("M", int=True, positive=True), N = symbols("N", int=True, positive=True), cfg = HilbertParameter(symbolic = True), logger = logger)
    logger.print(HilbertLogger.Verbosity.INFO, f"Nash-Gleichgewicht bei M Kunden und N Anbietern:")
    logger.print(HilbertLogger.Verbosity.INFO, f"    Verkaufspreis  v_sne = {v_sne}\n    Menge/Anbieter m_sne = {m_sne}\n    Nutzen         U_sne = {U_sne}")

    # Nash-Gleichgewicht an konkreten Beispielen
    cfg = HilbertParameter(symbolic = False)
    v_sne, m_sne, U_sne = nash_equilibrium_one_deviant(M = symbols("M", int=True, positive=True), N = symbols("N", int=True, positive=True), cfg = cfg, logger = logger)
    logger.print(HilbertLogger.Verbosity.INFO, f"Nash-Gleichgewicht bei M Kunden, N Anbietern und Beispielwerten:")
    logger.print(HilbertLogger.Verbosity.INFO, f"    Verkaufspreis  v_sne = {v_sne}\n    Menge/Anbieter m_sne = {m_sne}\n    Nutzen         U_sne = {U_sne}\n")
    for MN in [ (1,1), (1,1000000000), (1000000000,1000000000), (1000000000000,1000000000) ]:
        v_sne, m_sne, U_sne = nash_equilibrium_one_deviant(M = MN[0], N = MN[1], cfg = cfg, logger = logger)
        logger.print(HilbertLogger.Verbosity.INFO, f"Nash-Gleichgewicht bei {MN[0]} Kunden, {MN[1]} Anbietern und Beispielwerten: Verkaufspreis v_sne = {v_sne:.4f}, Menge/Anbieter m_sne = {m_sne:.4f}, Nutzen U_sne = {U_sne:.4f}")

    # Nash-Gleichgewicht an konkreten Beispielen, klassisch berechnet über N variable Preise (für kleine N zum Vergleich)
    for MN in [ (1,1), (2,2), (3,3), (10,10) ]:
        v_sne, m_sne, U_sne = nash_equilibrium_classical(M = MN[0], N = MN[1], cfg = cfg, logger = logger)
        v_sne_chk, m_sne_chk, U_sne_chk = nash_equilibrium_one_deviant(M = MN[0], N = MN[1], cfg = cfg, logger = logger)
        assert(math.isclose(v_sne, v_sne_chk, rel_tol=1e-6, abs_tol=1e-6) and math.isclose(m_sne, m_sne_chk, rel_tol=1e-6, abs_tol=1e-6) and math.isclose(U_sne, U_sne_chk, rel_tol=1e-6, abs_tol=1e-6))
        logger.print(HilbertLogger.Verbosity.INFO, f"Nash-Gleichgewicht bei {MN[0]} Kunden, {MN[1]} Anbietern und Beispielwerten: v_sne = {v_sne:.4f}, m_sne = {m_sne:.4f}, U_sne = {U_sne:.4f}  (klassisch berechnet)")

    # Nash-Gleichgewicht in Grenzwerten N -> oo:
    logger.print(HilbertLogger.Verbosity.INFO, f"\nNash-Gleichgewichte (Verkaufspreis v) bei N -> oo:\n")
    M, N = symbols("M N", int=True, positive=True)
    v_sne, m_sne, U_sne = nash_equilibrium_one_deviant(M = M, N = N, cfg = HilbertParameter(symbolic = True), logger = logger) # v_sne(M,N) = (M*N*h*v_0 + M*N*m_0 + M*e_n*h*s + M*e_n*h + N**2*e_n*r*s + N**2*e_n*r - N*e_n*r*s - N*e_n*r)/(M*N*h + M*h + N**2*r - N*r)
    v_sne_eg, m_sne_eg, U_sne_eg = nash_equilibrium_one_deviant(M = M, N = N, cfg = cfg, logger = logger) # e.g.: v_sne_eg(M,N) = 1.19*(0.126050420168067*M*N + 0.025*M + 1.0*N**2 - 1.0*N)/(0.025*M*N + 0.025*M + 1.0*N**2 - 1.0*N)
    logger.print(HilbertLogger.Verbosity.INFO, f"v_sne(M,N) = {v_sne}, e.g. v_sne_eg(M,N) = {v_sne_eg}")
    
    # Nash-Gleichgewicht bei M<<N, M,N->oo:
    v_sne_N_inf = limit(v_sne, N, oo) # N -> oo: N^2 dominiert => v_sne(M<<N, M,N->oo) = (e_n*r*s + e_n*r)/r
    v_sne_N_inf_eg = limit(v_sne_eg, N, oo) # e.g.: v_sne_eg(M<<N, M,N->oo) = 119/100
    logger.print(HilbertLogger.Verbosity.INFO, f"v_sne(M<<N, M,N -> oo) = {v_sne_N_inf}, e.g. v_sne_eg(M<<N, M,N -> oo) = {v_sne_N_inf_eg}")
    
    # Nash-Gleichgewicht bei M=N, N -> oo:
    v_sne_MeqN_N_inf = limit(v_sne.subs(M, N), N, oo) # M=N, N->oo: v_sne(M=N, M,N->oo) = (e_n*r*s + e_n*r + h*v_0 + m_0)/(h + r)
    v_sne_MeqN_N_inf_eg = limit(v_sne_eg.subs(M, N), N, oo) # e.g. v_sne_eg(M=N, M,N->oo) = 268/205
    logger.print(HilbertLogger.Verbosity.INFO, f"v_sne(M=N, M,N -> oo) = {v_sne_MeqN_N_inf}, e.g. v_sne_eg(M=N, M,N -> oo) = {v_sne_MeqN_N_inf_eg}")
    
    # Nash-Gleichgewicht bei festem Verhältnis M = MNR*N, MNR=M/N, N->oo:
    MNR = symbols("MNR", real=True)
    v_sne_MNR_N_inf = simplify(limit(v_sne.subs(M, N*MNR), N, oo)) # MNR=M/N, N->oo: v_sne(MNR=M/N, N->oo) = (MNR*h*v_0 + MNR*m_0 + e_n*r*s + e_n*r)/(MNR*h + r)
    v_sne_MNR_N_inf_eg = limit(v_sne_eg.subs(M, N*MNR), N, oo) # e.g. v_sne_eg(MNR=M/N, N->oo) = 2*(15*MNR + 119)/(5*(MNR + 40))
    logger.print(HilbertLogger.Verbosity.INFO, f"\nv_sne(MNR=M/N, N->oo) = {v_sne_MNR_N_inf}, e.g. v_sne_eg(MNR=M/N, N->oo) = {v_sne_MNR_N_inf_eg}")
    m_sne_MNR_N_inf = simplify(limit(m_sne.subs(M, N*MNR), N, oo)) # MNR=M/N, N->oo: m_sne(MNR=M/N, N->oo) = (MNR*h*v_0 + MNR*m_0 + e_n*r*s + e_n*r)/(MNR*h + r)
    m_sne_MNR_N_inf_eg = limit(m_sne_eg.subs(M, N*MNR), N, oo) # e.g. m_sne_eg(MNR=M/N, N->oo) = 2*(15*MNR + 119)/(5*(MNR + 40))
    logger.print(HilbertLogger.Verbosity.INFO, f"m_sne(MNR=M/N, N->oo) = {m_sne_MNR_N_inf}, e.g. m_sne_eg(MNR=M/N, N->oo) = {m_sne_MNR_N_inf_eg}")
    U_sne_MNR_N_inf = simplify(limit(U_sne.subs(M, N*MNR), N, oo)) # MNR=M/N, N->oo: U_sne(MNR=M/N, N->oo) = (MNR*h*v_0 + MNR*m_0 + e_n*r*s + e_n*r)/(MNR*h + r)
    U_sne_MNR_N_inf_eg = limit(U_sne_eg.subs(M, N*MNR), N, oo) # e.g. U_sne_eg(MNR=M/N, N->oo) = 2*(15*MNR + 119)/(5*(MNR + 40))
    logger.print(HilbertLogger.Verbosity.INFO, f"U_sne(MNR=M/N, N->oo) = {U_sne_MNR_N_inf}, e.g. U_sne_eg(MNR=M/N, N->oo) = {U_sne_MNR_N_inf_eg}\n")

    # Randbedingung: m_sne(MNR) >= 0, sonst verkaufen die Tankstellen negative Mengen und die Kunden streiken. Rechnerisch könnte dies bei sehr teuren Tankstellen geschehen, 
    # da die Verkaufsmenge umgekehrt proportional zum Verkaufspreis ist. Aus der Randbedingung m_sne(MNR) >= 0 folgt ein minimal erforderliches M/N, hier: M/N >= 0
    MNR_min_m_eq_0 = solve(Eq(m_sne_MNR_N_inf, 0), MNR) # MNR_min_m_eq_0 = MNR for m_sne(MNR) = 0
    MNR_min_m_eq_0_eg = solve(Eq(m_sne_MNR_N_inf_eg, 0), MNR) # MNR_min_m_eq_0_eg = MNR for m_sne(MNR) = 0 für die Beispielwerte
    MNR_min_m_eq_0_egf = [ mnr_min.evalf() for mnr_min in MNR_min_m_eq_0_eg if mnr_min.is_scalar ]
    MNR_min_m_eq_0_egf = min(MNR_min_m_eq_0_egf) if len(MNR_min_m_eq_0_egf) > 0 else 0
    logger.print(HilbertLogger.Verbosity.INFO, f"Min. M/N for m[n] >= 0 (N->oo): MNR_min = MNR|m=0 = {MNR_min_m_eq_0}, eg. MNR_min = MNR|m=0 = {MNR_min_m_eq_0_egf}")
    
    # Randbedingung: U_sne(MNR) >= 0, sonst übersteigen die laufenden Kosten den Nutzen und der Anbieter schliesst die Tankstelle. 
    # Aus der Randbedingung U_sne(MNR) >= 0 folgt ein minimal erforderliches M/N. Wird die kritische M/N-Schwelle unterschritten,
    # schliessen Tankstellen mit Verlusten, so dass M/N gross genug wird, um die Randbedingung einzuhalten. Im Grenzübergang M,N -> oo bleibt 
    # es natürlich trotzdem bei abzählbar unendlich vielen Hilbert-Tankstellen, nur eine Mindestanzahl Kunden pro Tankstelle wird eingehalten.
    # solve(U_sne_MNR_N_inf > 0, MNR) or reduce_inequalities(U_sne_MNR_N_inf > 0, MNR) cannot be solved using solve_univariate_inequality.
    MNR_min_U_eq_0 = solve(Eq(U_sne_MNR_N_inf, 0), MNR) # MNR_min_U_eq_0 = MNR for U_sne(MNR) = 0
    MNR_min_U_eq_0_eg = solve(Eq(U_sne_MNR_N_inf_eg, 0), MNR) # MNR_min_U_eq_0_eg = MNR for U_sne(MNR) = 0 für die Beispielwerte
    # solve liefert 2 Lösungen einer quadratischen Gleichung, eine Lösung MNR_min_U_eq_0_eg < 0 und eine Lösung MNR_min_U_eq_0_eg > 0. Nur MNR_min_U_eq_0_eg > 0 ist die gesuchte Lösung für MNR_min = MNR|U>=0.
    MNR_min_U_eq_0 = [ simplify(mnr_min) for mnr_min in MNR_min_U_eq_0 ]
    MNR_min_U_eq_0_eg = [ simplify(mnr_min) for mnr_min in MNR_min_U_eq_0_eg ]
    MNR_min_U_eq_0_egf = [ mnr_min.evalf() for mnr_min in MNR_min_U_eq_0_eg if mnr_min.is_scalar ]
    MNR_min_U_eq_0_egf = [ mnr_min for mnr_min in MNR_min_U_eq_0_egf if mnr_min >= 0 ]
    MNR_min_U_eq_0_egf = min(MNR_min_U_eq_0_egf) if len(MNR_min_U_eq_0_egf) > 0 else 0
    logger.print(HilbertLogger.Verbosity.INFO, f"Min. M/N for U[n] >= 0 (N->oo): MNR_min = MNR|U=0 = {MNR_min_U_eq_0}, eg. MNR_min = MNR|U=0 = {MNR_min_U_eq_0_egf}\n")

    # Nash-Gleichgewicht bei M/N->0, N->oo:
    v_sne_MNR_0_N_inf = simplify(limit(v_sne_MNR_N_inf, MNR, 0)) # MNR->0, N->oo: v_sne(MNR->0, N->oo)
    v_sne_MNR_0_N_inf_eg = limit(v_sne_MNR_N_inf_eg, MNR, 0) # e.g v_sne(MNR->0, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"v_sne(M/N->0, N->oo) = {v_sne_MNR_0_N_inf}, e.g. v_sne_eg(M/N->0, N->oo) = {v_sne_MNR_0_N_inf_eg}")
    m_sne_MNR_0_N_inf = simplify(limit(m_sne_MNR_N_inf, MNR, 0)) # MNR->0, N->oo: m_sne(MNR->0, N->oo)
    m_sne_MNR_0_N_inf_eg = limit(m_sne_MNR_N_inf_eg, MNR, 0) # e.g m_sne(MNR->0, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"m_sne(M/N->0, N->oo) = {m_sne_MNR_0_N_inf}, e.g. m_sne_eg(M/N->0, N->oo) = {m_sne_MNR_0_N_inf_eg}")
    U_sne_MNR_0_N_inf = simplify(limit(U_sne_MNR_N_inf, MNR, 0)) # MNR->0, N->oo: U_sne(MNR->0, N->oo)
    U_sne_MNR_0_N_inf_eg = limit(U_sne_MNR_N_inf_eg, MNR, 0) # e.g U_sne(MNR->0, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"U_sne(M/N->0, N->oo) = {U_sne_MNR_0_N_inf}, e.g. U_sne_eg(M/N->0, N->oo) = {U_sne_MNR_0_N_inf_eg}\n")

    # Nash-Gleichgewicht bei M/N=MNR_min_eg, N->oo (min. M/N für Randbedingungen m_sne(MNR) >= 0 und U_sne(MNR) >= 0 im Beispiel):
    MNR_min_eg = max(MNR_min_m_eq_0_egf, MNR_min_U_eq_0_egf)
    v_sne_MNR_min_N_inf = simplify(limit(v_sne_MNR_N_inf, MNR, MNR_min_eg)) # MNR=MNR_min_eg, N->oo: v_sne(MNR=MNR_min_eg, N->oo)
    v_sne_MNR_min_N_inf_eg = limit(v_sne_MNR_N_inf_eg, MNR, MNR_min_eg) # e.g v_sne(MNR=MNR_min_eg, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"v_sne(M/N={MNR_min_eg:.4f}, N->oo) = {v_sne_MNR_min_N_inf}, e.g. v_sne_eg(M/N={MNR_min_eg:.4f}, N->oo) = {v_sne_MNR_min_N_inf_eg}")
    m_sne_MNR_min_N_inf = simplify(limit(m_sne_MNR_N_inf, MNR, MNR_min_eg)) # MNR=MNR_min_eg, N->oo: m_sne(MNR=MNR_min_eg, N->oo)
    m_sne_MNR_min_N_inf_eg = limit(m_sne_MNR_N_inf_eg, MNR, MNR_min_eg) # e.g m_sne(MNR=MNR_min_eg, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"m_sne(M/N={MNR_min_eg:.4f}, N->oo) = {m_sne_MNR_min_N_inf}, e.g. m_sne_eg(M/N={MNR_min_eg:.4f}, N->oo) = {m_sne_MNR_min_N_inf_eg}")
    U_sne_MNR_min_N_inf = simplify(limit(U_sne_MNR_N_inf, MNR, MNR_min_eg)) # MNR=MNR_min_eg, N->oo: U_sne(MNR=MNR_min_eg, N->oo)
    U_sne_MNR_min_N_inf_eg = limit(U_sne_MNR_N_inf_eg, MNR, MNR_min_eg) # e.g U_sne(MNR=MNR_min_eg, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"U_sne(M/N={MNR_min_eg:.4f}, N->oo) = {U_sne_MNR_min_N_inf}, e.g. U_sne_eg(M/N={MNR_min_eg:.4f}, N->oo) = {U_sne_MNR_min_N_inf_eg}\n")

    # Nash-Gleichgewicht bei M/N=1, N->oo:
    v_sne_MNR_1_N_inf = simplify(limit(v_sne_MNR_N_inf, MNR, 1)) # MNR=1, N->oo: v_sne(MNR=1, N->oo)
    v_sne_MNR_1_N_inf_eg = limit(v_sne_MNR_N_inf_eg, MNR, 1) # e.g v_sne(MNR=1, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"v_sne(M/N=1, N->oo) = {v_sne_MNR_1_N_inf}, e.g. v_sne_eg(M/N=1, N->oo) = {v_sne_MNR_1_N_inf_eg}")
    m_sne_MNR_1_N_inf = simplify(limit(m_sne_MNR_N_inf, MNR, 1)) # MNR=1, N->oo: m_sne(MNR=1, N->oo)
    m_sne_MNR_1_N_inf_eg = limit(m_sne_MNR_N_inf_eg, MNR, 1) # e.g m_sne(MNR=1, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"m_sne(M/N=1, N->oo) = {m_sne_MNR_1_N_inf}, e.g. m_sne_eg(M/N=1, N->oo) = {m_sne_MNR_1_N_inf_eg}")
    U_sne_MNR_1_N_inf = simplify(limit(U_sne_MNR_N_inf, MNR, 1)) # MNR=1, N->oo: U_sne(MNR=1, N->oo)
    U_sne_MNR_1_N_inf_eg = limit(U_sne_MNR_N_inf_eg, MNR, 1) # e.g U_sne(MNR=1, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"U_sne(M/N=1, N->oo) = {U_sne_MNR_1_N_inf}, e.g. U_sne_eg(M/N=1, N->oo) = {U_sne_MNR_1_N_inf_eg}\n")

    # Nash-Gleichgewicht bei M/N->oo, N->oo:
    v_sne_MNR_inf_N_inf = simplify(limit(v_sne_MNR_N_inf, MNR, oo)) # MNR->oo, N->oo: v_sne(MNR->oo, N->oo)
    v_sne_MNR_inf_N_inf_eg = limit(v_sne_MNR_N_inf_eg, MNR, oo) # e.g v_sne(MNR->oo, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"v_sne(M/N->oo, N->oo) = {v_sne_MNR_inf_N_inf}, e.g. v_sne_eg(M/N->oo, N->oo) = {v_sne_MNR_inf_N_inf_eg}")
    m_sne_MNR_inf_N_inf = simplify(limit(m_sne_MNR_N_inf, MNR, oo)) # MNR->oo, N->oo: m_sne(MNR->oo, N->oo)
    m_sne_MNR_inf_N_inf_eg = limit(m_sne_MNR_N_inf_eg, MNR, oo) # e.g m_sne(MNR->oo, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"m_sne(M/N->oo, N->oo) = {m_sne_MNR_inf_N_inf}, e.g. m_sne_eg(M/N->oo, N->oo) = {m_sne_MNR_inf_N_inf_eg}")
    U_sne_MNR_inf_N_inf = simplify(limit(U_sne_MNR_N_inf, MNR, oo)) # MNR->oo, N->oo: U_sne(MNR->oo, N->oo)
    U_sne_MNR_inf_N_inf_eg = limit(U_sne_MNR_N_inf_eg, MNR, oo) # e.g U_sne(MNR->oo, N->oo)
    logger.print(HilbertLogger.Verbosity.INFO, f"U_sne(M/N->oo, N->oo) = {U_sne_MNR_inf_N_inf}, e.g. U_sne_eg(M/N->oo, N->oo) = {U_sne_MNR_inf_N_inf_eg}\n")

    # Plot Gleichgewichtspreis v über M/N für N->oo und Beispielwerte, i.e. plot v_sne_MNR_N_inf_eg über MNR
    plt1 = plot((v_sne_MNR_N_inf_eg, (MNR,0,1000)), title="Gleichgewichtspreis v über M/N für N->oo (Beispielwerte)", xlabel="M/N", ylabel="v", axis_center=(0,cfg.e_n), legend=False, show=False)
    # Plot Gleichgewichtspreis v über M und N für Beispielwerte, i.e. plot3d v_sne_eg über M und N
    plt2 = plot3d((v_sne_eg, (N,1,50), (M,1,1000)), title="Gleichgewichtspreis v über M und N (Beispielwerte)", xlabel="N", ylabel="M", zlabel="v", use_cm=True, legend=False, show=False)
    # Plot v über N für M=1, 10, 100, 1000
    v_sne_plt3 = [ nash_equilibrium_one_deviant(M = M, N = N, cfg = cfg, logger = logger)[0] for M in [1, 10, 100, 1000] ]
    plt3 = plot((v_sne_plt3[0],{"label": "M=1"}), (v_sne_plt3[1],{"label": "M=10"}), (v_sne_plt3[2],{"label": "M=100"}), (v_sne_plt3[3],{"label": "M=1000"}), (N,1,100), 
                title="Gleichgewichtspreis v über N für M=1,10,100,1000 (Beispielwerte)", xlabel="N", ylabel="v", axis_center=(0,cfg.e_n), legend=True, show=False)
    pltg = plotgrid(plt1, plt3, plt2, nc=2, nr=2, show=False)
    pltg.show()

if __name__ == "__main__":
    nash_equilibrium_hilbert()
    # nash_equilibrium_hilbert_with_fees()
