"""
Berechnung der perfekten Bayes-Gleichgewichte (Perfect Bayes Equilibrium, PBE) am Beispiel zweier konkurrierender, nicht-kooperativer Tankstellen.

Eine Tankstelle habe ein Monopol, ein Konkurrent erwägt den Markteintritt. Der Monopolist und der Konkurrent können mit derselben Wahrscheinlichkeit entwedder reich (R) oder arm (A) sein.
Tritt der Konkurrent in den Markt ein (e), kann der Monopolist aggressiv (a) mit Konkurrenzkampf oder friedlich (f) reagieren. Reagiert der Monopolist friedlich, teilen sich beide den Markt.
Reagiert der Monopolist aggressiv, kann der Konkurrent ihn herausfordern (h) oder aufgeben und den Konkurrenzkampf verlieren. Da ein Konkurrenzkampf Geld kostet, wird der Monopolist nur so wenig wie nötig bluffen, 
um den Konkurrent vom Markteintritt abzuhalten.

Das 1-Konkurrentenspiel ⁠$(N,A,T,q,u)$ ist definiert durch:
* Menge der Spieler: $N$ = { $K$ (Konkurrent), $M$ (Monopolist) }
* Menge der Aktionen: $A_1$ = { $e$ (Eintritt), $n$ (Kein Eintritt) }, $A_2$ = { $a$ (aggressiv), $f$ (Friedlich) }, $A_1|a$ = { $h$ (herausfordern), $v$ (verlieren) }
* Menge der Typen: $T_1$ = { $R$ (Reich), $A$ (arm) }, $T_2$ = { $R$ (Reich), $A$ (arm) }
* Wahrscheinlichkeiten: $P_M(M=R) = q$, $P(M=A) = 1 - q$, $P_K(K=R) = q$, $P(K=A) = 1 - q$
* Nutzenfunktionen: $(u1, u2)$ nach Spielbaum (Harsanyi-Transformation): siehe ../docs/pbe_gas_stations.svg
    * Kein Markteintritt: $(u_1, u_2) = (0, 18)$
    * Markteintritt und friedliche Marktteilung: $(u_1, u_2) = (2, 9)$
    * Markteintritt, aggressiver reicher Monopolist, reicher Konkurrent gibt auf: $(u_1, u_2) = (-5, 10)$
    * Markteintritt, aggressiver reicher Monopolist, armer Konkurrent gibt (sofort und nach geringen Verlusten) auf: $(u_1, u_2) = (-3, 8)$
    * Markteintritt, aggressiver armer Monopolist, reicher Konkurrent gibt (nach Zögern) auf: $(u_1, u_2) = (-7, 17)$
    * Markteintritt, aggressiver armer Monopolist, armer Konkurrent gibt (zügig) auf: $(u_1, u_2) = (-4, 17)$
    * Markteintritt, aggressiver reicher Monopolist, reicher Konkurrent kämpft → beide verlieren: $(u_1, u_2) = (-6, -1)$
    * Markteintritt, aggressiver reicher Monopolist, armer Konkurrent kämpft → Konkurrent verliert: $(u_1, u_2) = (-6, 17)$
    * Markteintritt, aggressiver armer Monopolist, reicher Konkurrent kämpft → Konkurrent gewinnt: $(u_1, u_2) = (13, 11)$
    * Markteintritt, aggressiver armer Monopolist, armer Konkurrent kämpft → Konkurrent gewinnt: $(u_1, u_2) = (13, -1)$

Eine ausführlichere Beschreibung und Erläuterungen sind in Beispiel 7 (Markteintritt, PBE) in SpieltheorieEinfuehrung.md.
"""
import numpy as np
from sympy import *
from sympy.assumptions.relation.binrel import AppliedBinaryRelation
from sympy.solvers.inequalities import reduce_rational_inequalities
from pbe_two_gas_stations_check import *

# Return the symbolic probabilities q = P(M=R) = P(K=R), PaMR = P(a|M=R), PaMA = P(a|M=A), PeKR = P(e|K=R), PeKA = P(e|K=A), PKRe = P(K=R|e), PMRa = P(M=R|a), PMRf = P(M=R|f), PhKR = P(h|K=R), PhKA = P(h|K=A)
def get_symbols():
    q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA = symbols("q P(a|M=R) P(a|M=A) P(e|K=R) P(e|K=A) P(K=R|e) P(M=R|a) P(M=R|f) P(h|K=R) P(h|K=A)", real=True)
    return [ q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA ]

# Setup all payoffs
def setup_payoff_functions():

    # payoff functions (u1,u2) from final nodes of the game tree:  
    u = {} # dict of payoff functions
    u["K=R,n"]         = ( 0, 18) # (u1, u2) | K=R,n
    u["K=A,n"]         = ( 0, 18) # (u1, u2) | K=A,n

    u["K=R,e,M=R,f"]   = ( 2,  9) # (u1, u2) | K=R,e,M=R,f # ( 4,  9)
    u["K=R,e,M=A,f"]   = ( 2,  9) # (u1, u2) | K=R,e,M=A,f # ( 4,  9)
    u["K=A,e,M=R,f"]   = ( 2,  9) # (u1, u2) | K=R,e,M=R,f # ( 4,  9)
    u["K=A,e,M=A,f"]   = ( 2,  9) # (u1, u2) | K=R,e,M=A,f # ( 4,  9)

    u["K=R,e,M=R,a,h"] = (-6, -1) # (u1, u2) | K=R,e,M=R,a,h
    u["K=R,e,M=R,a,v"] = (-5, 10) # (u1, u2) | K=R,e,M=R,a,v
    u["K=R,e,M=A,a,h"] = (13, 11) # (u1, u2) | K=R,e,M=A,a,h # (13, -1) 
    u["K=R,e,M=A,a,v"] = (-7, 17) # (u1, u2) | K=R,e,M=A,a,v
    u["K=A,e,M=R,a,h"] = (-6, 17) # (u1, u2) | K=A,e,M=R,a,h
    u["K=A,e,M=R,a,v"] = (-3,  8) # (u1, u2) | K=A,e,M=R,a,v
    u["K=A,e,M=A,a,h"] = (13, -1) # (u1, u2) | K=A,e,M=A,a,h
    u["K=A,e,M=A,a,v"] = (-4, 17) # (u1, u2) | K=A,e,M=A,a,v

    print(f"Endknoten:")
    for key in u.keys():
        print(f"    (u1, u2) | {key} = {u[key]}")
    print(f"\nNutzenfunktionen:")

    # symbolic probabilities q = P(M=R) = P(K=R), PaMR = P(a|M=R), PaMA = P(a|M=A), PeKR = P(e|K=R), PeKA = P(e|K=A), PKRe = P(K=R|e), PMRa = P(M=R|a), PMRf = P(M=R|f), PhKR = P(h|K=R), PhKA = P(h|K=A)
    q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA = get_symbols() # If a, f, e are on-path, probabilities PMRa = P(M=R|a), PMRf = P(M=R|f) and PKRe = P(K=R|e) are later replaced using Bayes

    # Nutzenfunktionen u1 des Konkurrenten
    e = {} # dict of payoff expressions
    e['u1(n|K=R)'] = S(u["K=R,n"][0]) # u1(n|K=R)
    e['u1(n|K=A)'] = S(u["K=A,n"][0]) # u1(n|K=A)
    e['u1(h|a,K=R)'] = S((u["K=R,e,M=R,a,h"][0])*PMRa + (u["K=R,e,M=A,a,h"][0])*(1-PMRa)) # u1(h|a,K=R)
    e['u1(h|a,K=A)'] = S((u["K=A,e,M=R,a,h"][0])*PMRa + (u["K=A,e,M=A,a,h"][0])*(1-PMRa)) # u1(h|a,K=A)
    e['u1(v|a,K=R)'] = S((u["K=R,e,M=R,a,v"][0])*PMRa + (u["K=R,e,M=A,a,v"][0])*(1-PMRa)) # u1(v|a,K=R)
    e['u1(v|a,K=A)'] = S((u["K=A,e,M=R,a,v"][0])*PMRa + (u["K=A,e,M=A,a,v"][0])*(1-PMRa)) # u1(v|a,K=A)
    e['h>v|a,K=R'] = S(e['u1(h|a,K=R)'] > e['u1(v|a,K=R)']) # Der reiche Konkurrent wählt h|a (a ist on-path) gdw. u1(h|a,K=R) > u1(v|a,K=R)
    e['h>v|a,K=A'] = S(e['u1(h|a,K=A)'] > e['u1(v|a,K=A)']) # Der  arme  Konkurrent wählt h|a (a ist on-path) gdw. u1(h|a,K=A) > u1(v|a,K=A)
    print(f"    u1(n|K=R) = {simplify(e['u1(n|K=R)'])}")
    print(f"    u1(n|K=A) = {simplify(e['u1(n|K=A)'])}")
    print(f"    u1(h|a,K=R) = {simplify(e['u1(h|a,K=R)'])}")
    print(f"    u1(h|a,K=A) = {simplify(e['u1(h|a,K=A)'])}")
    print(f"    u1(v|a,K=R) = {simplify(e['u1(v|a,K=R)'])}")
    print(f"    u1(v|a,K=A) = {simplify(e['u1(v|a,K=A)'])}")
    print(f"    u1(h|a,K=R) > u1(v|a,K=R) <=> {simplify(e['h>v|a,K=R'])}")
    print(f"    u1(h|a,K=A) > u1(v|a,K=A) <=> {simplify(e['h>v|a,K=A'])}")

    # Nutzenfunktionen u2 des Monopolisten
    e['u2|M=R,K=R,e'] = Piecewise((u["K=R,e,M=R,a,h"][1], e['h>v|a,K=R']), (u["K=R,e,M=R,a,v"][1], True)) # u2|M=R,K=R,e
    e['u2|M=R,K=A,e'] = Piecewise((u["K=A,e,M=R,a,h"][1], e['h>v|a,K=A']), (u["K=A,e,M=R,a,v"][1], True)) # u2|M=R,K=A,e
    e['u2|M=A,K=R,e'] = Piecewise((u["K=R,e,M=A,a,h"][1], e['h>v|a,K=R']), (u["K=R,e,M=A,a,v"][1], True)) # u2|M=A,K=R,e
    e['u2|M=A,K=A,e'] = Piecewise((u["K=A,e,M=A,a,h"][1], e['h>v|a,K=A']), (u["K=A,e,M=A,a,v"][1], True)) # u2|M=A,K=A,e
    e['u2(a|M=R)'] = (e['u2|M=R,K=R,e'])*(PKRe) + (e['u2|M=R,K=A,e'])*(1 - PKRe) # u2(a|M=R)
    e['u2(a|M=A)'] = (e['u2|M=A,K=R,e'])*(PKRe) + (e['u2|M=A,K=A,e'])*(1 - PKRe) # u2(a|M=A)
    e['u2(f|M=R)'] = (u["K=R,e,M=R,f"][1])*(PKRe) + (u["K=A,e,M=R,f"][1])*(1 - PKRe) # u2(f|M=R)
    e['u2(f|M=A)'] = (u["K=R,e,M=A,f"][1])*(PKRe) + (u["K=A,e,M=A,f"][1])*(1 - PKRe) # u2(f|M=A)
    print(f"    u2(a|K=R,M=R) = {simplify(e['u2|M=R,K=R,e'])}")
    print(f"    u2(a|K=A,M=R) = {simplify(e['u2|M=R,K=A,e'])}")
    print(f"    u2(a|K=R,M=A) = {simplify(e['u2|M=A,K=R,e'])}")
    print(f"    u2(a|K=A,M=A) = {simplify(e['u2|M=A,K=A,e'])}")
    print(f"    u2(a|M=R) = {simplify(e['u2(a|M=R)'])}")
    print(f"    u2(a|M=A) = {simplify(e['u2(a|M=A)'])}")
    print(f"    u2(f|M=R) = {simplify(e['u2(f|M=R)'])}")
    print(f"    u2(f|M=A) = {simplify(e['u2(f|M=A)'])}")

    # u2-Nutzen für Konsistenzbedingung bei friedlicher Reaktion (a off-path)
    e['u2|M=R,K=R,f'] = Piecewise((u["K=R,e,M=R,a,h"][1], e['h>v|a,K=R']), (u["K=R,e,M=R,a,v"][1], True)) # u2|M=R,K=R,e wenn a off-path
    e['u2|M=R,K=A,f'] = Piecewise((u["K=A,e,M=R,a,h"][1], e['h>v|a,K=A']), (u["K=A,e,M=R,a,v"][1], True)) # u2|M=R,K=A,e wenn a off-path
    e['u2|M=A,K=R,f'] = Piecewise((u["K=R,e,M=A,a,h"][1], e['h>v|a,K=R']), (u["K=R,e,M=A,a,v"][1], True)) # u2|M=A,K=R,e wenn a off-path
    e['u2|M=A,K=A,f'] = Piecewise((u["K=A,e,M=A,a,h"][1], e['h>v|a,K=A']), (u["K=A,e,M=A,a,v"][1], True)) # u2|M=A,K=A,e wenn a off-path
    e['u2(af|M=R)'] = (e['u2|M=R,K=R,f'])*(PKRe) + (e['u2|M=R,K=A,f'])*(1 - PKRe) # u2(a|M=R) für Konsistenzbedingung bei friedlicher Reaktion (a off-path)
    e['u2(af|M=A)'] = (e['u2|M=A,K=R,f'])*(PKRe) + (e['u2|M=A,K=A,f'])*(1 - PKRe) # u2(a|M=A) für Konsistenzbedingung bei friedlicher Reaktion (a off-path)

    # Ex ante entry-Nutzen für Indifferenzgleichungen:
    e['u1|M=R,K=R,a'] = Piecewise((u["K=R,e,M=R,a,h"][0], e['h>v|a,K=R']), (u["K=R,e,M=R,a,v"][0], True)) # u1|M=R,K=R,a
    e['u1|M=A,K=R,a'] = Piecewise((u["K=R,e,M=A,a,h"][0], e['h>v|a,K=R']), (u["K=R,e,M=A,a,v"][0], True)) # u1|M=A,K=R,a
    e['u1|M=R,K=A,a'] = Piecewise((u["K=A,e,M=R,a,h"][0], e['h>v|a,K=A']), (u["K=A,e,M=R,a,v"][0], True)) # u1|M=R,K=A,a
    e['u1|M=A,K=A,a'] = Piecewise((u["K=A,e,M=A,a,h"][0], e['h>v|a,K=A']), (u["K=A,e,M=A,a,v"][0], True)) # u1|M=A,K=A,a
    e['u1_ante(e|K=R)'] = e['u1|M=R,K=R,a']*q*PaMR + e['u1|M=A,K=R,a']*(1-q)*PaMA + (u["K=R,e,M=R,f"][0])*q*(1-PaMR) + (u["K=R,e,M=A,f"][0])*(1-q)*(1-PaMA) # u1_ante(e|K=R)
    e['u1_ante(e|K=A)'] = e['u1|M=R,K=A,a']*q*PaMR + e['u1|M=A,K=A,a']*(1-q)*PaMA + (u["K=A,e,M=R,f"][0])*q*(1-PaMR) + (u["K=A,e,M=A,f"][0])*(1-q)*(1-PaMA) # u1_ante(e|K=A)
    print(f"    u1_ante(e|K=R) = {simplify(e['u1_ante(e|K=R)'])}")
    print(f"    u1_ante(e|K=A) = {simplify(e['u1_ante(e|K=A)'])}")

    # Ex post Nutzen nach Signal für PBE-Konsistenzprüfung:
    e['u1_post(e|K=R)'] = e['u1|M=R,K=R,a']*PMRa + (u["K=R,e,M=R,f"][0])*(PMRf) + e['u1|M=A,K=R,a']*(1-PMRa) + (u["K=R,e,M=A,f"][0])*(1-PMRf) # u1_post(e|K=R)
    e['u1_post(e|K=A)'] = e['u1|M=R,K=A,a']*PMRa + (u["K=A,e,M=R,f"][0])*(PMRf) + e['u1|M=A,K=A,a']*(1-PMRa) + (u["K=A,e,M=A,f"][0])*(1-PMRf) # u1_post(e|K=A)
    print(f"    u1_post(e|K=R) = {simplify(e['u1_post(e|K=R)'])}")
    print(f"    u1_post(e|K=A) = {simplify(e['u1_post(e|K=A)'])}")

    # Indifferenzgleichungen
    print(f"\nIndifferenzgleichungen:")
    print(f"    Der reiche Konkurrent waehlt h|a gdw. u1(h|a,K=R) > u1(v|a,K=R) <=> {simplify(e['u1(h|a,K=R)'] > e['u1(v|a,K=R)'])}")
    print(f"    Der  arme  Konkurrent waehlt h|a gdw. u1(h|a,K=A) > u1(v|a,K=A) <=> {simplify(e['u1(h|a,K=A)'] > e['u1(v|a,K=A)'])}")
    print(f"    Der reiche Konkurrent waehlt e|K=R gdw. u1(e|K=R,ante) > u1(n|K=R) <=> {simplify(e['u1_ante(e|K=R)'] > e['u1(n|K=R)'])}")
    print(f"    Der  arme  Konkurrent waehlt e|K=A gdw. u1(e|K=A,ante) > u1(n|K=A) <=> {simplify(e['u1_ante(e|K=A)'] > e['u1(n|K=A)'])}")
    print(f"    Der reiche Konkurrent ist mit e|K=R ex post PBE-konsistent gdw. u1(e|K=R,post) > u1(n|K=R) <=> {simplify(e['u1_post(e|K=R)'] > e['u1(n|K=R)'])}")
    print(f"    Der  arme  Konkurrent ist mit e|K=A ex post PBE-konsistent gdw. u1(e|K=A,post) > u1(n|K=A) <=> {simplify(e['u1_post(e|K=A)'] > e['u1(n|K=A)'])}")
    print(f"    Der reiche Monopolist waehlt a gdw. u2(a|M=R) > u2(f|M=R) <=> {simplify(e['u2(a|M=R)'] > e['u2(f|M=R)'])}")
    print(f"    Der  arme  Monopolist waehlt a gdw. u2(a|M=A) > u2(f|M=A) <=> {simplify(e['u2(a|M=A)'] > e['u2(f|M=A)'])}")

    # Bayes
    print(f"\nBayes:")
    print(f"    P(M=R|a) = q * P(a|M=R) / (q * P(a|M=R) + (1-q) * P(a|M=A))")
    print(f"    P(M=R|f) = q * (1-P(a|M=R)) / (q * (1-P(a|M=R)) + (1-q) * (1-P(a|M=A)))")
    print(f"    P(K=R|e) = q * P(e|K=R) / (q * P(e|K=R) + (1-q) * P(e|K=A))\n")
    return e

# Replace P(M=R|a) using Bayes: P(M=R|a) = q * P(a|M=R) / (q * P(a|M=R) + (1-q) * P(a|M=A))
def apply_bayes_PMRa(e, do_piecewise = False):
    q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA = get_symbols()
    if(do_piecewise):
        PMRa_bayes = Piecewise(((q * PaMR) / (q * PaMR + (1-q) * PaMA), (q * PaMR + (1-q) * PaMA) > 0), (0, True))
    else:
        PMRa_bayes = (q * PaMR) / (q * PaMR + (1-q) * PaMA)
    e = e.subs(PMRa, PMRa_bayes)
    return e

# Replace P(M=R|f) using Bayes: P(M=R|f) = q * (1-P(a|M=R)) / (q * (1-P(a|M=R)) + (1-q) * (1-P(a|M=A)))
def apply_bayes_PMRf(e, do_piecewise = False):
    q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA = get_symbols()
    if(do_piecewise):
        PMRf_bayes = Piecewise(((q * (1-PaMR)) / (q * (1-PaMR) + (1-q) * (1-PaMA)), (q * (1-PaMR) + (1-q) * (1-PaMA)) > 0), (0, True))
    else:
        PMRf_bayes = (q * (1-PaMR)) / (q * (1-PaMR) + (1-q) * (1-PaMA))
    e = e.subs(PMRf, PMRf_bayes)
    return e

# Replace P(K=R|e) using Bayes: P(K=R|e) = q * P(e|K=R) / (q * P(e|K=R) + (1-q) * P(e|K=A))
def apply_bayes_PKRe(e, do_piecewise = False):
    q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA = get_symbols()
    if(do_piecewise):
        PKRe_bayes = Piecewise(((q * PeKR) / (q * PeKR + (1-q) * PeKA), (q * PeKR + (1-q) * PeKA) > 0), (0, True))
    else:
        PKRe_bayes = (q * PeKR) / (q * PeKR + (1-q) * PeKA)
    e = e.subs(PKRe, PKRe_bayes)
    return e

# Revert Bayes-formulas q * P(a|M=R) / (q * P(a|M=R) + (1-q) * P(a|M=A)) = P(M=R|a), 
# q * (1-P(a|M=R)) / (q * (1-P(a|M=R)) + (1-q) * (1-P(a|M=A))) = P(M=R|f) 
# und q * P(e|K=R) / (q * P(e|K=R) + (1-q) * P(e|K=A)) = P(K=R|e)
def unapply_bayes(e):
    q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA = get_symbols()
    PMRa_bayes = (q * PaMR) / (q * PaMR + (1-q) * PaMA)
    PMRf_bayes = (q * (1-PaMR)) / (q * (1-PaMR) + (1-q) * (1-PaMA))
    PKRe_bayes = (q * PeKR) / (q * PeKR + (1-q) * PeKA)
    e = e.subs(PMRa_bayes, PMRa)
    e = e.subs(simplify(-1*PMRa_bayes), -PMRa)
    e = e.subs(PMRf_bayes, PMRf)
    e = e.subs(simplify(-1*PMRf_bayes), -PMRf)
    e = e.subs(PKRe_bayes, PKRe)
    e = e.subs(simplify(-1*PKRe_bayes), -PKRe)
    return e

# Prüft die Erfüllbarkeit der von satisfiable ermittelten Belegung durch
# 1. Ersetzen der Piecewise-Ausdrücke und Konvertierung in die disjunktive Normalform (ODER-Verknüpfung),
# 2. In jedem einzelnen Konjunktionsterm: Reduktion der Intervalle, 
#    Einsetzen der rechten Seiten bei Gleichheit von Wahrscheinlichkeitssymbolen,
#    und Suche nach einer Belegung (Beispielwerte, Zeuge), die die Bedingung erfüllt.
# Liefert den kleinsten gefundenen Ausdruck zurück, der äquivalent zu <sat_expr> ist.
def pbe_satisfiable_check(title, sat_expr):
    if sat_expr is False:
        print(f"{title}: PBE check <=> False (nicht erfuellbar)")
        return sat_expr
    if sat_expr is True:
        print(f"{title}: PBE check <=> True (immer erfuellbar)")
        return sat_expr
    # Replace (Piecewise((expr1,cond),(expr2,True)) <= rhs) by ((cond & (expr1<=rhs)) | (~cond & (expr2<=rhs))) 
    atoms = True
    for atom, atom_value in sat_expr.items():
        if atom_value != True:
            print(f"{title}: PBE check <=> nicht erfuellbar, {atom}: {atom_value}")
            return False # DNF is not satisfiable if one sub-expression is not satisfiable
        atom = replace_piecewise(atom)
        atoms = And(atoms, atom)
    # Convert to DNF and check each single term
    sat_expr_dnf = to_dnf(atoms)
    assert(isinstance(sat_expr_dnf, Or))
    sat_expr_reduced = False
    symbols_list = get_symbols()
    for dnf_atom in sat_expr_dnf.args:
        assert(isinstance(dnf_atom, And))
        atom_reduced = reduce_intervals(dnf_atom.args)
        if atom_reduced == False:
            continue
        if atom_reduced == True:
            sat_expr_reduced = True
            break
        assert(isinstance(atom_reduced, And))
        # Check that equalities in atom_reduced can be solved using sympy.solve
        # and replace equalities with their solution
        cnf_equalities = [cnf_atom for cnf_atom in atom_reduced.args if cnf_atom.rel_op == Eq.rel_op]
        if len(cnf_equalities) > 0:
            cnf_equalities_solved = solve(cnf_equalities, symbols_list)
            if len(cnf_equalities_solved) == 0:
                continue # equalities have no solution, thus atom_reduced is not solvable
            if len(cnf_equalities_solved) > 0:
                assert(len(cnf_equalities_solved[0]) == len(symbols_list))
                for n, var in enumerate(symbols_list):
                    atom_reduced.subs(var, cnf_equalities_solved[0][n])
            # After replacing all equalities with their solution we need to verify the inequalities
            cnf_inequalities = [cnf_atom for cnf_atom in atom_reduced.args if cnf_atom.rel_op != Eq.rel_op]
            atom_reduced = True
            for cnf_atom in cnf_inequalities:
                atom_reduced = And(cnf_atom, atom_reduced)
        # Check that inequalities in atom_reduced can be solved using sympy.reduce_inequalities
        for var in symbols_list:
            try:
                if atom_reduced.has(var) and reduce_inequalities(atom_reduced.args, var) == False:
                    atom_reduced = False
                    break
            except NotImplementedError as exc:
                pass
        if atom_reduced == False:
            continue
        # atom_reduced is solvable in principle, now let's try to find a witness
        sat_expr_reduced = Or(sat_expr_reduced, atom_reduced)
        # atom_reduced = unapply_bayes(atom_reduced)
        cnf_atoms = True
        for cnf_atom in atom_reduced.args:
            cnf_atoms = And(cnf_atoms, simplify(cnf_atom))
        atom_reduced = cnf_atoms
        grid_check_success, grid_check_witness, grid_check_msg = pbe_grid_check(atom_reduced, symbols_list)
        if grid_check_success == True:
            print(f"{title}: PBE check <=> True, {atom_reduced} erfuellt durch {grid_check_witness}")
            print(f"{title}: {grid_check_msg}")
            return grid_check_witness # witness found, pbe approved
    print(f"{title}: PBE check <=> {sat_expr_reduced}")
    return sat_expr_reduced

# Evaluierung PBE mit einer gegebenen Konsistenzbedingung, welche aus der Strategie (pooling, separating mit reiner oder gemischter Strategie) folgt
def pbe_consistency(title, cond_consistency, do_bayes_PMRa = False, do_bayes_PMRf = False, do_bayes_PKRe = False):
    q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA = get_symbols()
    cond_consistency_simple = simplify(cond_consistency)
    pbe_consistent = cond_consistency_simple
    if do_bayes_PMRa: # Note: simplify can throw an invalid NaN comparison exception after applying Bayes rules
        pbe_consistent = apply_bayes_PMRa(pbe_consistent, False)
    if do_bayes_PMRf:
        pbe_consistent = apply_bayes_PMRf(pbe_consistent, False)
    if do_bayes_PKRe:
        pbe_consistent = apply_bayes_PKRe(pbe_consistent, False)
    cond_probabilities = True
    if pbe_consistent.has(q):
        cond_probabilities = cond_probabilities & (q >= 0) & (q <= 1)
    if pbe_consistent.has(PaMR):
        cond_probabilities = cond_probabilities & (PaMR >= 0) & (PaMR <= 1)
    if pbe_consistent.has(PaMA):
        cond_probabilities = cond_probabilities & (PaMA >= 0) & (PaMA <= 1)
    if pbe_consistent.has(PeKR):
        cond_probabilities = cond_probabilities & (PeKR >= 0) & (PeKR <= 1)
    if pbe_consistent.has(PeKA):
        cond_probabilities = cond_probabilities & (PeKA >= 0) & (PeKA <= 1)
    if pbe_consistent.has(PKRe):
        cond_probabilities = cond_probabilities & (PKRe >= 0) & (PKRe <= 1)
    if pbe_consistent.has(PMRa):
        cond_probabilities = cond_probabilities & (PMRa >= 0) & (PMRa <= 1)
    if pbe_consistent.has(PMRf):
        cond_probabilities = cond_probabilities & (PMRf >= 0) & (PMRf <= 1)
    if pbe_consistent.has(PhKR):
        cond_probabilities = cond_probabilities & (PhKR >= 0) & (PhKR <= 1)
    if pbe_consistent.has(PhKA):
        cond_probabilities = cond_probabilities & (PhKA >= 0) & (PhKA <= 1)
    pbe_condition = pbe_consistent & cond_probabilities
    pbe_satisfiable = satisfiable(pbe_condition)
    if isinstance(pbe_satisfiable, dict):
        pbe_satisfiable = dict(sorted(pbe_satisfiable.items(), key=lambda kv: str(kv[0]))) # sort dictionary pbe_satisfiable by its keys for better debugging and readability
    print(f"{title}: PBE konsistent <=> {pbe_consistent}")
    print(f"{title}: PBE erfuellbar <=> {pbe_satisfiable}")
    if pbe_satisfiable == True:
        print(f"{title}: PBE immer konsistent")
    elif pbe_satisfiable == False:
        print(f"{title}: PBE niemals konsistent")
    else:
        symbols_list = get_symbols()
        if not do_bayes_PMRa and not do_bayes_PMRf and not do_bayes_PKRe:
            grid_check_success, grid_check_witness, grid_check_msg = pbe_grid_check(pbe_consistent, symbols_list) # search for pbe example values
        else: # grid check incl. bayes is too timeconsuming
            grid_check_success, grid_check_witness, grid_check_msg = pbe_grid_check(cond_consistency_simple, symbols_list) # fast plausibility check excl. bayes
            if grid_check_success == True: # use the fast plausibility check as an initial guess for a grid check
                pbe_candidate = pbe_consistent
                for atom in grid_check_witness.args:
                    if atom.rel_op == Eq.rel_op:
                        pbe_candidate = pbe_candidate.subs(atom.lhs, atom.rhs)
                grid_check_success, grid_check_witness, grid_check_msg = pbe_grid_check(pbe_candidate, symbols_list) # grid check of pbe candidate incl. bayes
        if grid_check_success:
            print(f"{title}: {grid_check_msg}")
        else:
            pbe_satisfiable_check(title, pbe_satisfiable) # check pbe consistency by converting to DNF and single term analysation
    print(f"")

# Evaluierung Pooling-PBE's. Pooling Strategien sind:
# | Nr. | Reicher Konkurrent | Armer Konkurrent | Reicher Monopolist | Armer Monopolist |
# |  1  |        n           |         n        |         -          |       -          |
# |  2  |        e           |         e        |         f          |       f          |
# |  3  |       e,h          |        e,h       |         a          |       a          |
# |  4  |       e,v          |        e,v       |         a          |       a          |
# Pooling auf $n$: $a$ und $f$ sind beide off-path ⇒ Randbedingung: $0 ≤ P(M=R|a), P(M=R|f) ≤ 1$.
# Reiche und arme Monopolisten wählen nach Eintritt immer $a$: $a$ ist on-path, $f$ ist off-path ⇒ Randbedingung: $P(M=R|a)$ nach Bayes, $0 ≤ P(M=R|f) ≤ 1$
# Reiche und arme Monopolisten wählen nach Eintritt immer $f$: $a$ ist off-path, $f$ ist on-path ⇒ Randbedingung: $0 ≤ P(M=R|a) ≤ 1$, $P(M=R|f)$ nach Bayes
# Konkurrent tritt ein: e ist on-path ⇒ Randbedingung: P(K=R|e) nach Bayes (Konkurrent tritt nicht ein: n ist on-path, hat aber keine Folgeentscheidungen)
# Nutzen u2(a|M=R) bzw. u2(a|M=A) für Konsistenzbedingung bei aggressiver Reaktion (a ist on-path)
# Nutzen u2(af|M=R) bzw. u2(af|M=A) für Konsistenzbedingung bei friedlicher Reaktion (a ist off-path)
def pbe_pooling(e):
    # 1. Pooling auf n: Konkurrent wählt immer n (nicht eintreten)
    #    PBE ist konsistent, gdw. u1(n|K=R) >= u1_post(e|K=R) & u1(n|K=A) >= u1_post(e|K=A)
    pbe_consistency("Pooling auf n (Konkurrent: niemals eintreten)",
                    And(e['u1(n|K=R)'] >= e['u1_post(e|K=R)'], e['u1(n|K=A)'] >= e['u1_post(e|K=A)']),
                    do_bayes_PMRa = False, do_bayes_PMRf = False, do_bayes_PKRe = False)
    # 2. Pooling auf e,f: Konkurrent wählt immer e (eintreten), Monopolist wählt immer f (friedlich)
    #    PBE ist konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(f|M=R) >= u2(af|M=R) & u2(f|M=A) >= u2(af|M=A)
    pbe_consistency("Pooling auf e,f (Konkurrent: immer eintreten, Monopolist: immer friedlich)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(f|M=R)'] >= e['u2(af|M=R)'], e['u2(f|M=A)'] >= e['u2(af|M=A)']),
                    do_bayes_PMRa = False, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 3. Pooling auf e,a,h: Konkurrent wählt immer e (eintreten), Monopolist wählt immer a (aggressiv), Konkurrent wählt immer h (herausfordern)
    #    PBE ist konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(h|a,K=R) >= u1(v|a,K=R) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Pooling auf e,a,h (Konkurrent: immer eintreten, Monopolist: immer aggressiv, Konkurrent: immer herausfordernd)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = False, do_bayes_PKRe = True)
    # 4. Pooling auf e,a,v: Konkurrent wählt immer e (eintreten), Monopolist wählt immer a (aggressiv), Konkurrent wählt immer v (verlieren)
    #    PBE ist konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(v|a,K=R) >= u1(h|a,K=R) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Pooling auf e,a,v (Konkurrent: immer eintreten, Monopolist: immer aggressiv, Konkurrent: immer verlierend)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = False, do_bayes_PKRe = True)

# Evaluierung Separating-PBE's mit reinen Stategien. Reine Separating-Strategien sind:
# | Nr. | Reicher Konkurrent | Armer Konkurrent | Reicher Monopolist | Armer Monopolist |
# | --- | ------------------ | ---------------- | ------------------ | ---------------- |
# |   1 |        e           |         n        |         f          |       f          |
# |   2 |       e,h          |         n        |         a          |       f          |
# |   3 |       e,v          |         n        |         a          |       f          |
# |   4 |       e,h          |         n        |         f          |       a          |
# |   5 |       e,v          |         n        |         f          |       a          |
# |   6 |       e,h          |         n        |         a          |       a          |
# |   7 |       e,v          |         n        |         a          |       a          |
# |   8 |        n           |         e        |         f          |       f          |
# |   9 |        n           |        e,h       |         a          |       f          |
# |  10 |        n           |        e,v       |         a          |       f          |
# |  11 |        n           |        e,h       |         f          |       a          |
# |  12 |        n           |        e,v       |         f          |       a          |
# |  13 |        n           |        e,h       |         a          |       a          |
# |  14 |        n           |        e,v       |         a          |       a          |
# |  15 |       e,h          |        e,h       |         a          |       f          |
# |  16 |       e,h          |        e,v       |         a          |       f          |
# |  17 |       e,v          |        e,h       |         a          |       f          |
# |  18 |       e,v          |        e,v       |         a          |       f          |
# |  19 |       e,h          |        e,h       |         f          |       a          |
# |  20 |       e,h          |        e,v       |         f          |       a          |
# |  21 |       e,v          |        e,h       |         f          |       a          |
# |  22 |       e,v          |        e,v       |         f          |       a          |
# |  23 |       e,h          |        e,v       |         a          |       a          |
# |  24 |       e,v          |        e,h       |         a          |       a          |
# Reiche und arme Monopolisten wählen nach Eintritt immer $a$: $a$ ist on-path, $f$ ist off-path ⇒ Randbedingung: $P(M=R|a)$ nach Bayes, $0 ≤ P(M=R|f) ≤ 1$
# Reiche und arme Monopolisten wählen nach Eintritt immer $f$: $a$ ist off-path, $f$ ist on-path ⇒ Randbedingung: $0 ≤ P(M=R|a) ≤ 1$, $P(M=R|f)$ nach Bayes
# Reiche und arme Monopolisten wählen nach Eintritt unterschiedlich $a$ oder $f$: $a$ und $f$ sind on-path ⇒ Randbedingung: $P(M=R|a), P(M=R|f)$ nach Bayes.
# Konkurrent tritt ein: e ist on-path ⇒ Randbedingung: P(K=R|e) nach Bayes (Konkurrent tritt nicht ein: n ist on-path, hat aber keine Folgeentscheidungen)
# Nutzen u2(a|M=R) bzw. u2(a|M=A) für Konsistenzbedingung bei aggressiver Reaktion (a ist on-path)
# Nutzen u2(af|M=R) bzw. u2(af|M=A) für Konsistenzbedingung bei friedlicher Reaktion (a ist off-path)
def pbe_separating_pure_strategies(e):
    # 1. e|K=R, n|K=A, f|M=R, f|M=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1(n|K=A) >= u1_post(e|K=A) & u2(f|M=R) >= u2(af|M=R) & u2(f|M=A) >= u2(af|M=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, n|K=A, f|M=R, f|M=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1(n|K=A)'] >= e['u1_post(e|K=A)'], e['u2(f|M=R)'] >= e['u2(af|M=R)'], e['u2(f|M=A)'] >= e['u2(af|M=A)']),
                    do_bayes_PMRa = False, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 2. e|K=R, n|K=A, a|M=R, f|M=A, h|K=R : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1(n|K=A) >= u1_post(e|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(f|M=A) >= u2(a|M=A) & u1(h|a,K=R) >= u1(v|a,K=R)
    pbe_consistency("Separating (reine Strategie: e|K=R, n|K=A, a|M=R, f|M=A, h|K=R)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1(n|K=A)'] >= e['u1_post(e|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(f|M=A)'] >= e['u2(a|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 3. e|K=R, n|K=A, a|M=R, f|M=A, v|K=R : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1(n|K=A) >= u1_post(e|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(f|M=A) >= u2(a|M=A) & u1(v|a,K=R) >= u1(h|a,K=R)
    pbe_consistency("Separating (reine Strategie: e|K=R, n|K=A, a|M=R, f|M=A, v|K=R)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1(n|K=A)'] >= e['u1_post(e|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(f|M=A)'] >= e['u2(a|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 4. e|K=R, n|K=A, f|M=R, a|M=A, h|K=R : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1(n|K=A) >= u1_post(e|K=A) & u2(f|M=R) >= u2(a|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(h|a,K=R) >= u1(v|a,K=R)
    pbe_consistency("Separating (reine Strategie: e|K=R, n|K=A, f|M=R, a|M=A, h|K=R)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1(n|K=A)'] >= e['u1_post(e|K=A)'], e['u2(f|M=R)'] >= e['u2(a|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 5. e|K=R, n|K=A, f|M=R, a|M=A, v|K=R : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1(n|K=A) >= u1_post(e|K=A) & u2(f|M=R) >= u2(a|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(v|a,K=R) >= u1(h|a,K=R)
    pbe_consistency("Separating (reine Strategie: e|K=R, n|K=A, f|M=R, a|M=A, v|K=R)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1(n|K=A)'] >= e['u1_post(e|K=A)'], e['u2(f|M=R)'] >= e['u2(a|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 6. e|K=R, n|K=A, a|M=R, a|M=A, h|K=R : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1(n|K=A) >= u1_post(e|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(h|a,K=R) >= u1(v|a,K=R)
    pbe_consistency("Separating (reine Strategie: e|K=R, n|K=A, a|M=R, a|M=A, h|K=R)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1(n|K=A)'] >= e['u1_post(e|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = False, do_bayes_PKRe = True)
    # 7. e|K=R, n|K=A, a|M=R, a|M=A, v|K=R : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1(n|K=A) >= u1_post(e|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(v|a,K=R) >= u1(h|a,K=R)
    pbe_consistency("Separating (reine Strategie: e|K=R, n|K=A, a|M=R, a|M=A, v|K=R)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1(n|K=A)'] >= e['u1_post(e|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = False, do_bayes_PKRe = True)
    # 8. e|K=A, n|K=R, f|M=R, f|M=A : PBE konsistent, gdw. u1_ante(e|K=A) >= u1(n|K=A) & u1(n|K=R) >= u1_post(e|K=R) & u2(f|M=R) >= u2(af|M=R) & u2(f|M=A) >= u2(af|M=A)
    pbe_consistency("Separating (reine Strategie: e|K=A, n|K=R, f|M=R, f|M=A)",
                    And(e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u1(n|K=R)'] >= e['u1_post(e|K=R)'], e['u2(f|M=R)'] >= e['u2(af|M=R)'], e['u2(f|M=A)'] >= e['u2(af|M=A)']),
                    do_bayes_PMRa = False, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 9. e|K=A, n|K=R, a|M=R, f|M=A, h|K=A : PBE konsistent, gdw. u1_ante(e|K=A) >= u1(n|K=A) & u1(n|K=R) >= u1_post(e|K=R) & u2(a|M=R) >= u2(f|M=R) & u2(f|M=A) >= u2(a|M=A) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=A, n|K=R, a|M=R, f|M=A, h|K=A)",
                    And(e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u1(n|K=R)'] >= e['u1_post(e|K=R)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(f|M=A)'] >= e['u2(a|M=A)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 10. e|K=A, n|K=R, a|M=R, f|M=A, v|K=A : PBE konsistent, gdw. u1_ante(e|K=A) >= u1(n|K=A) & u1(n|K=R) >= u1_post(e|K=R) & u2(a|M=R) >= u2(f|M=R) & u2(f|M=A) >= u2(a|M=A) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=A, n|K=R, a|M=R, f|M=A, v|K=A)",
                    And(e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u1(n|K=R)'] >= e['u1_post(e|K=R)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(f|M=A)'] >= e['u2(a|M=A)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 11. e|K=A, n|K=R, f|M=R, a|M=A, h|K=A : PBE konsistent, gdw. u1_ante(e|K=A) >= u1(n|K=A) & u1(n|K=R) >= u1_post(e|K=R) & u2(f|M=R) >= u2(a|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=A, n|K=R, f|M=R, a|M=A, h|K=A)",
                    And(e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u1(n|K=R)'] >= e['u1_post(e|K=R)'], e['u2(f|M=R)'] >= e['u2(a|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 12. e|K=A, n|K=R, f|M=R, a|M=A, v|K=A : PBE konsistent, gdw. u1_ante(e|K=A) >= u1(n|K=A) & u1(n|K=R) >= u1_post(e|K=R) & u2(f|M=R) >= u2(a|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=A, n|K=R, f|M=R, a|M=A, v|K=A)",
                    And(e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u1(n|K=R)'] >= e['u1_post(e|K=R)'], e['u2(f|M=R)'] >= e['u2(a|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 13. e|K=A, n|K=R, a|M=R, a|M=A, h|K=A : PBE konsistent, gdw. u1_ante(e|K=A) >= u1(n|K=A) & u1(n|K=R) >= u1_post(e|K=R) & u2(a|M=R) >= u2(f|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=A, n|K=R, a|M=R, a|M=A, h|K=A)",
                    And(e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u1(n|K=R)'] >= e['u1_post(e|K=R)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = False, do_bayes_PKRe = True)
    # 14. e|K=A, n|K=R, a|M=R, a|M=A, v|K=A : PBE konsistent, gdw. u1_ante(e|K=A) >= u1(n|K=A) & u1(n|K=R) >= u1_post(e|K=R) & u2(a|M=R) >= u2(f|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=A, n|K=R, a|M=R, a|M=A, v|K=A)",
                    And(e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u1(n|K=R)'] >= e['u1_post(e|K=R)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = False, do_bayes_PKRe = True)
    # 15. e|K=R, e|K=A, a|M=R, f|M=A, h|K=R, h|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(f|M=A) >= u2(a|M=A) & u1(h|a,K=R) >= u1(v|a,K=R) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, a|M=R, f|M=A, h|K=R, h|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(f|M=A)'] >= e['u2(a|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 16. e|K=R, e|K=A, a|M=R, f|M=A, h|K=R, v|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(f|M=A) >= u2(a|M=A) & u1(h|a,K=R) >= u1(v|a,K=R) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, a|M=R, f|M=A, h|K=R, v|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(f|M=A)'] >= e['u2(a|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 17. e|K=R, e|K=A, a|M=R, f|M=A, v|K=R, h|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(f|M=A) >= u2(a|M=A) & u1(v|a,K=R) >= u1(h|a,K=R) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, a|M=R, f|M=A, v|K=R, h|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(f|M=A)'] >= e['u2(a|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 18. e|K=R, e|K=A, a|M=R, f|M=A, v|K=R, v|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(f|M=A) >= u2(a|M=A) & u1(v|a,K=R) >= u1(h|a,K=R) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, a|M=R, f|M=A, v|K=R, v|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(f|M=A)'] >= e['u2(a|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 19. e|K=R, e|K=A, f|M=R, a|M=A, h|K=R, h|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(f|M=R) >= u2(a|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(h|a,K=R) >= u1(v|a,K=R) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, f|M=R, a|M=A, h|K=R, h|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(f|M=R)'] >= e['u2(a|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 20. e|K=R, e|K=A, f|M=R, a|M=A, h|K=R, v|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(f|M=R) >= u2(a|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(h|a,K=R) >= u1(v|a,K=R) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, f|M=R, a|M=A, h|K=R, v|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(f|M=R)'] >= e['u2(a|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 21. e|K=R, e|K=A, f|M=R, a|M=A, v|K=R, h|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(f|M=R) >= u2(a|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(v|a,K=R) >= u1(h|a,K=R) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, f|M=R, a|M=A, v|K=R, h|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(f|M=R)'] >= e['u2(a|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 22. e|K=R, e|K=A, f|M=R, a|M=A, v|K=R, v|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(f|M=R) >= u2(a|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(v|a,K=R) >= u1(h|a,K=R) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, f|M=R, a|M=A, v|K=R, v|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(f|M=R)'] >= e['u2(a|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 23. e|K=R, e|K=A, a|M=R, a|M=A, h|K=R, v|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(h|a,K=R) >= u1(v|a,K=R) & u1(v|a,K=A) >= u1(h|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, a|M=R, a|M=A, h|K=R, v|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(h|a,K=R)'] >= e['u1(v|a,K=R)'], e['u1(v|a,K=A)'] >= e['u1(h|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = False, do_bayes_PKRe = True)
    # 24. e|K=R, e|K=A, a|M=R, a|M=A, v|K=R, h|K=A : PBE konsistent, gdw. u1_ante(e|K=R) >= u1(n|K=R) & u1_ante(e|K=A) >= u1(n|K=A) & u2(a|M=R) >= u2(f|M=R) & u2(a|M=A) >= u2(f|M=A) & u1(v|a,K=R) >= u1(h|a,K=R) & u1(h|a,K=A) >= u1(v|a,K=A)
    pbe_consistency("Separating (reine Strategie: e|K=R, e|K=A, a|M=R, a|M=A, v|K=R, h|K=A)",
                    And(e['u1_ante(e|K=R)'] >= e['u1(n|K=R)'], e['u1_ante(e|K=A)'] >= e['u1(n|K=A)'], e['u2(a|M=R)'] >= e['u2(f|M=R)'], e['u2(a|M=A)'] >= e['u2(f|M=A)'], e['u1(v|a,K=R)'] >= e['u1(h|a,K=R)'], e['u1(h|a,K=A)'] >= e['u1(v|a,K=A)']),
                    do_bayes_PMRa = True, do_bayes_PMRf = False, do_bayes_PKRe = True)

# Evaluierung Separating-PBE's mit semi-gemischten Stategien (d.h. armer oder reicher Typ mischt, aber nicht beide). Semi-gemischte Separating-Strategien sind (Auswahl):
# | Nr. | Reicher Konkurrent | Armer Konkurrent  | Reicher Monopolist | Armer Monopolist |
# | --- | ------------------ | ----------------- | ------------------ | ---------------- |
# |   1 |        e,h         |        e,h        |      P(a|M=R)      |        f         |
# |   2 |        e,h         |        e,h        |      P(a|M=R)      |        a         |
# |   3 |        e,h         |        e,h        |         f          |     P(a|M=A)     |
# |   4 |        e,h         |        e,h        |         a          |     P(a|M=A)     |
# |   5 |        e,h         |        e,v        |      P(a|M=R)      |        f         |
# |   6 |        e,h         |        e,v        |      P(a|M=R)      |        a         |
# |   7 |        e,h         |        e,v        |         f          |     P(a|M=A)     |
# |   8 |        e,h         |        e,v        |         a          |     P(a|M=A)     |
# |   9 |        e,v         |        e,h        |      P(a|M=R)      |        f         |
# |  10 |        e,v         |        e,h        |      P(a|M=R)      |        a         |
# |  11 |        e,v         |        e,h        |         f          |     P(a|M=A)     |
# |  12 |        e,v         |        e,h        |         a          |     P(a|M=A)     |
# |  13 | P(e|K=R),h         |        e,h        |      P(a|M=R)      |        f         |
# |  14 | P(e|K=R),h         |        e,h        |      P(a|M=R)      |        a         |
# |  15 | P(e|K=R),h         |        e,h        |         f          |     P(a|M=A)     |
# |  16 | P(e|K=R),h         |        e,h        |         a          |     P(a|M=A)     |
# |  17 |        e,h         |        e,P(e|K=R) |      P(a|M=R)      |        f         |
# |  18 |        e,h         |        e,P(e|K=R) |      P(a|M=R)      |        a         |
# |  19 |        e,h         |        e,P(e|K=R) |         f          |     P(a|M=A)     |
# |  20 |        e,h         |        e,P(e|K=R) |         a          |     P(a|M=A)     |
# Monopolist mischt mit 0 < P(a|M=A) < 1 ⇒ a und f immer on-path ⇒ Randbedingung: P(M=R|a), P(M=R|f) nach Bayes.
# Konkurrent tritt ein: e ist on-path ⇒ Randbedingung: P(K=R|e) nach Bayes (Konkurrent tritt nicht ein: n ist on-path, hat aber keine Folgeentscheidungen)
# Reicher Monopolist wählt immer a ⇒ u2(a|M=R) >= u2(f|M=R), reicher Monopolist wählt immer f ⇒ u2(f|M=R) >= u2(a|M=R), reicher Monopolist mischt ⇒ u2(a|M=R) = u2(f|M=R).
# Armer Monopolist wählt immer a ⇒ u2(a|M=A) >= u2(f|M=A), armer Monopolist wählt immer f ⇒ u2(f|M=A) >= u2(a|M=A), armer Monopolist mischt ⇒ u2(a|M=A) = u2(f|M=A).
def pbe_separating_semi_mixed_strategies(e):
    # 1. e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=0, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=0, h|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(f|M=A)'], e['u2(a|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 2. e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=1, h|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 3. e|K=R, e|K=A, P(a|M=R)=0, 0<P(a|M=A)<1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, P(a|M=R)=0, 0<P(a|M=A)<1, h|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(f|M=R)'], e['u2(a|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 4. e|K=R, e|K=A, P(a|M=R)=1, 0<P(a|M=A)<1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, P(a|M=R)=1, 0<P(a|M=A)<1, h|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 5. e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=0, h|K=R, v|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=0, h|K=R, v|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(f|M=A)'], e['u2(a|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(v|a,K=A)'], e['u1(h|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 6. e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=1, h|K=R, v|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=1, h|K=R, v|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(v|a,K=A)'], e['u1(h|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 7. e|K=R, e|K=A, P(a|M=R)=0, 0<P(a|M=A)<1, h|K=R, v|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, P(a|M=R)=0, 0<P(a|M=A)<1, h|K=R, v|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(f|M=R)'], e['u2(a|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(v|a,K=A)'], e['u1(h|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 8. e|K=R, e|K=A, P(a|M=R)=1, 0<P(a|M=A)<1, h|K=R, v|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, P(a|M=R)=1, 0<P(a|M=A)<1, h|K=R, v|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(v|a,K=A)'], e['u1(h|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 9. e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=0, v|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=0, v|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(f|M=A)'], e['u2(a|M=A)']), 
                        Ge(e['u1(v|a,K=R)'], e['u1(h|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 10. e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=1, v|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=1, v|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(v|a,K=R)'], e['u1(h|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 11. e|K=R, e|K=A, P(a|M=R)=0, 0<P(a|M=A)<1, v|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, P(a|M=R)=0, 0<P(a|M=A)<1, v|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(f|M=R)'], e['u2(a|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(v|a,K=R)'], e['u1(h|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 12. e|K=R, e|K=A, P(a|M=R)=1, 0<P(a|M=A)<1, v|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, e|K=A, P(a|M=R)=1, 0<P(a|M=A)<1, v|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(v|a,K=R)'], e['u1(h|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 13. 0<P(e|K=R)<1, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=0, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: 0<P(e|K=R)<1, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=0, h|K=R, h|K=A)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(f|M=A)'], e['u2(a|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 14. 0<P(e|K=R)<1, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: 0<P(e|K=R)<1, e|K=A, 0<P(a|M=R)<1, P(a|M=A)=1, h|K=R, h|K=A)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 15. 0<P(e|K=R)<1, e|K=A, P(a|M=R)=0, 0<P(a|M=A)<1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: 0<P(e|K=R)<1, e|K=A, P(a|M=R)=0, 0<P(a|M=A)<1, h|K=R, h|K=A)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(f|M=R)'], e['u2(a|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 16. 0<P(e|K=R)<1, e|K=A, P(a|M=R)=1, 0<P(a|M=A)<1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: 0<P(e|K=R)<1, e|K=A, P(a|M=R)=1, 0<P(a|M=A)<1, h|K=R, h|K=A)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 17. e|K=R, 0<P(e|K=A)<1, 0<P(a|M=R)<1, P(a|M=A)=0, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, 0<P(e|K=A)<1, 0<P(a|M=R)<1, P(a|M=A)=0, h|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(f|M=A)'], e['u2(a|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 18. e|K=R, 0<P(e|K=A)<1, 0<P(a|M=R)<1, P(a|M=A)=1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, 0<P(e|K=A)<1, 0<P(a|M=R)<1, P(a|M=A)=1, h|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Ge(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 19. e|K=R, 0<P(e|K=A)<1, P(a|M=R)=0, 0<P(a|M=A)<1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, 0<P(e|K=A)<1, P(a|M=R)=0, 0<P(a|M=A)<1, h|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(f|M=R)'], e['u2(a|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 20. e|K=R, 0<P(e|K=A)<1, P(a|M=R)=1, 0<P(a|M=A)<1, h|K=R, h|K=A
    pbe_consistency("Separating (semi-gemischte Strategie: e|K=R, 0<P(e|K=A)<1, P(a|M=R)=1, 0<P(a|M=A)<1, h|K=R, h|K=A)",
                    And(Ge(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Ge(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)

# Evaluierung Separating-PBE's mit voll-gemischten Stategien (d.h. beide Typen mischen). Voll-gemischte Separating-Strategien sind (Auswahl):
# | Nr. | Reicher Konkurrent | Armer Konkurrent  | Reicher Monopolist | Armer Monopolist |
# | --- | ------------------ | ----------------- | ------------------ | ---------------- |
# |   1 | P(e|K=R),h         | P(e|K=A),h        |      P(a|M=R)      |     P(a|M=A)     |
# |   2 | P(e|K=R),h         | P(e|K=A),v        |      P(a|M=R)      |     P(a|M=A)     |
# |   3 | P(e|K=R),v         | P(e|K=A),h        |      P(a|M=R)      |     P(a|M=A)     |
# |   4 | P(e|K=R),v         | P(e|K=A),v        |      P(a|M=R)      |     P(a|M=A)     |
# |   5 | P(e|K=R),h         |         n         |      P(a|M=R)      |     P(a|M=A)     |
# |   6 | P(e|K=R),v         |         n         |      P(a|M=R)      |     P(a|M=A)     |
# |   7 |         n          | P(e|K=A),h        |      P(a|M=R)      |     P(a|M=A)     |
# |   8 |         n          | P(e|K=A),v        |      P(a|M=R)      |     P(a|M=A)     |
# |   9 | P(e|K=R),P(h|K=R)  | P(e|K=A),P(h|K=A) |      P(a|M=R)      |     P(a|M=A)     |
# Monopolist spielt immer mit 0 < P(a|M=A) < 1 ⇒ a und f immer on-path ⇒ Randbedingung: P(M=R|a), P(M=R|f) nach Bayes.
# Konkurrent tritt ein: e ist on-path ⇒ Randbedingung: P(K=R|e) nach Bayes (Konkurrent tritt nicht ein: n ist on-path, hat aber keine Folgeentscheidungen)
# Achtung Falle: e['u2(a|M=R)'] == e['u2(f|M=R)'] ist immer False (python object compare), Eq(e['u2(a|M=R)'], e['u2(f|M=R)']) ist eine sympy-Gleichung.
# Für Ungleichungen geht auch <, <=, >, >= statt Lt, Le, Gt, Ge; für sympy-Gleichungen ist Eq statt == erforderlich.
def pbe_separating_fully_mixed_strategies(e):
    # 1. e|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, h|K=R, h|K=A
    pbe_consistency("Separating (gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, h|K=R, h|K=A)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 2. e|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, h|K=R, v|K=A
    pbe_consistency("Separating (gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, h|K=R, v|K=A)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Ge(e['u1(v|a,K=A)'], e['u1(h|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 3. e|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, v|K=R, h|K=A
    pbe_consistency("Separating (gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, v|K=R, h|K=A)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(v|a,K=R)'], e['u1(h|a,K=R)']), Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 4. e|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, v|K=R, v|K=A
    pbe_consistency("Separating (gemischte Strategie: e|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, v|K=R, v|K=A)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(v|a,K=R)'], e['u1(h|a,K=R)']), Ge(e['u1(v|a,K=A)'], e['u1(h|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 5. e|K=R, n|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, h|K=R
    pbe_consistency("Separating (gemischte Strategie: e|K=R, n|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, h|K=R)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1(n|K=A)'], e['u1_post(e|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=R)'], e['u1(v|a,K=R)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 6. e|K=R, n|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, v|K=R
    pbe_consistency("Separating (gemischte Strategie: e|K=R, n|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, v|K=R)",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Ge(e['u1(n|K=A)'], e['u1_post(e|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(v|a,K=R)'], e['u1(h|a,K=R)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 7. n|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, h|K=A
    pbe_consistency("Separating (gemischte Strategie: n|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, h|K=A)",
                    And(Ge(e['u1(n|K=R)'], e['u1_post(e|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 8. n|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, v|K=A
    pbe_consistency("Separating (gemischte Strategie: n|K=R, e|K=A, 0<P(a|M=R)<1, 0<P(a|M=A)<1, v|K=A)",
                    And(Ge(e['u1(n|K=R)'], e['u1_post(e|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Ge(e['u1(v|a,K=A)'], e['u1(h|a,K=A)'])), 
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)
    # 9. 0<P(e|K=R)<1, 0<P(e|K=A)<1, 0<P(a|M=R)<1, 0<P(a|M=A)<1, 0<P(h|K=R)<1, 0<P(h|K=A)<1
    pbe_consistency("Separating (vollgemischte Strategie: 0<Pe|K=R)<1, 0<Pe|K=A)<1, 0<P(a|M=R)<1, 0<P(a|M=A)<1, 0<P(h|K=R)<1, 0<P(h|K=A)<1",
                    And(Eq(e['u1_ante(e|K=R)'], e['u1(n|K=R)']), Eq(e['u1_ante(e|K=A)'], e['u1(n|K=A)']), 
                        Eq(e['u2(a|M=R)'], e['u2(f|M=R)']), Eq(e['u2(a|M=A)'], e['u2(f|M=A)']), 
                        Eq(e['u1(h|a,K=R)'], e['u1(v|a,K=R)']), Eq(e['u1(h|a,K=A)'], e['u1(v|a,K=A)'])),
                    do_bayes_PMRa = True, do_bayes_PMRf = True, do_bayes_PKRe = True)

if __name__ == "__main__":
    print(f"PBE am Beispiel zweier Tankstellen (Konkurrent und Monopolist):\n")
    # Setup der Nutzenfunktionen
    e = setup_payoff_functions()
    # Pooling PBE's
    pbe_pooling(e)
    # Separating PBE's mit reinen Stategien
    pbe_separating_pure_strategies(e)
    # Separating-PBE's mit semi-gemischten Stategien
    pbe_separating_semi_mixed_strategies(e)
    # Separating-PBE's mit voll-gemischten Stategien (keine PBEs)
    # pbe_separating_fully_mixed_strategies(e)
