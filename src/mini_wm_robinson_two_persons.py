"""
Mini-WM: Ein minimalistisches Wirtschaftsmodell,
Teil 2: Robinson-Crusoe-Wirtschaft mit 2 Personen: Robinson und Freitag.

Freie Entscheidungsvariablen sind 20 Unbekannte (Robinson: i=1, Freitag: i=2):
Li_fisch (Arbeitseinsätze für Fischproduktion), Li_nuss (Arbeitseinsätze für Kokosnussproduktion), Li_subsistenz (Zeiten für Subsistenz), Li_fisch_research (Arbeitseinsätze für Forschung in Fischtechnologie), Li_nuss_research (Arbeitseinsätze für Forschung in Kokosnusstechnologie)
lambdai_fisch_invest (Wachstum Fischzucht-Kapital), lambdai_nuss_invest (Wachstum Kokosnuss-Kapital), 
wi_fisch (Subsistenz-Anteil am Fischkonsum), wi_nuss (Subsistenz-Anteil am Nusskonsum),
tau1_fisch (Robinsons Tauschanteil), tau2_nuss (Freitags Tauschanteil)

Grenzen der Entscheidungsvariablen:
0 <= Li_fisch, Li_nuss, Li_subsistenz, Li_fisch_research, Li_nuss_research <= L_max
0 <= lambdai_fisch_invest, lambdai_nuss_invest <= lambda_invest_max
0 <= tau1_fisch, tau2_nuss, wi_fisch, wi_nuss <= 1

Zeitrestriktion:  
Li_genuss(t) = L_max - Li_fisch - Li_nuss - Li_subsistenz - Li_fisch_invest(t) - Li_nuss_invest(t) - Li_fisch_research - Li_nuss_research # L_max = 24 Stunden
Randbedingungen: Li_genuss(t) >= 0
Li_fisch_invest(t) und Li_nuss_invest(t) gehen mit wachsendem t gegen 0, Li_genuss(t) steigt mit wachsendem t gegen (L_max - Li_fisch - Li_nuss - Li_subsistenz), während Li_fisch, Li_nuss, Li_subsistenz, Li_fisch_research und Li_nuss_research konstant bleiben.

Produktionsfunktionen:
Yi_fisch(t, Li_fisch) = Ai_fisch(Ri_fisch(t)) * Li_fisch^alphai_L_fisch * (Ki_fisch_basis + Ki_fisch(t))^alphai_K_fisch # Fischproduktion mit Anfangskapital Ki_fisch_basis
Yi_nuss(t, Li_nuss) = Ai_nuss(Ri_nuss(t)) * Li_nuss^alphai_L_nuss * (Ki_nuss_basis + Ki_nuss(t))^alphai_K_nuss          # Kokosnussproduktion mit Anfangskapital Ki_nuss_basis

Investitionen in Fischzucht:
Ki_fisch(t) = K_fisch_max * (1 - exp(-lambdai_fisch_invest * t))        # Fischzucht-Kapital
Ii_fisch(t) = Ai_fisch_invest * Ci_fisch_invest(t)^alphai_fisch_invest  # Investition in Fischzucht
Ii_fisch(t) = dKi_fisch(t)/dt = K_fisch_max * lambdai_fisch_invest * exp(-lambdai_fisch_invest * t) # Investition = dKapital/dt
Li_fisch_invest(t) = Ci_fisch_invest(t) / ri_fisch_invest # In Li_fisch_invest Zeiteinheiten werden Ci_fisch_invest Fische investiert
<=> Li_fisch_invest(t) = ((Ii_fisch(t)/Ai_fisch_invest)^(1/alphai_fisch_invest))/ri_fisch_invest = (((K_fisch_max * lambdai_fisch_invest*exp(-lambdai_fisch_invest*t))/Ai_fisch_invest)^(1/alphai_fisch_invest))/ri_fisch_invest

Investitionen in Kokosnussanbau:
Ki_nuss(t) = K_nuss_max * (1 - exp(-lambdai_nuss_invest * t))       # Kokosnuss-Kapital
Ii_nuss(t) = Ai_nuss_invest * Ci_nuss_invest(t)^alphai_nuss_invest  # Investition in Kokosnussanbau
Ii_nuss(t) = dKi_nuss(t)/dt = K_nuss_max * lambdai_nuss_invest * exp(-lambdai_nuss_invest * t) # Investition = dKapital/dt
Li_nuss_invest(t) = Ci_nuss_invest(t) / ri_nuss_invest # In Li_nuss_invest Zeiteinheiten werden Ci_nuss_invest Kokosnüsse neu gepflanzt.
<=> Li_nuss_invest(t) = ((Ii_nuss(t)/Ai_nuss_invest)^(1/alphai_nuss_invest))/ri_nuss_invest = (((K_nuss_max * lambdai_nuss_invest*exp(-lambdai_nuss_invest*t))/Ai_nuss_invest)^(1/alphai_nuss_invest))/ri_nuss_invest

Produktivität:
Ai_fisch(Ri_fisch(t)) = Ai_fisch_0 + gammai_fisch_research * Ri_fisch(t)
Ai_nuss(Ri_nuss(t)) = Ai_nuss_0 + gammai_nuss_research * Ri_nuss(t)

Forschung in der Minimalvariante durch reinen Arbeitseinsatz:
Ri_fisch(t) = (1 + (1-alphai_research) * Ai_fisch_research * Li_fisch_research * t)^(1/(1-alphai_research)) - 1 für 0 <= alphai_research < 1
Ri_fisch(t) = (Ai_fisch_research * Li_fisch_research + 1)^t - 1 für alphai_research = 1
Ri_nuss(t) = (1 + (1-alphai_research) * Ai_nuss_research * Li_nuss_research * t)^(1/(1-alphai_research)) - 1 für 0 <= alphai_research < 1
Ri_nuss(t) = (Ai_nuss_research * Li_nuss_research + 1)^t - 1 für alphai_research = 1

Güterverwendung = Subsistenzkonsum + Genusskonsum + Investition
Ci_fisch(t) = Ci_fisch_subsistenz(t) + Ci_fisch_genuss(t) + Ci_fisch_invest(t)  # Gesamte Fischverwendung (Konsum und Investition)
Ci_nuss(t) = Ci_nuss_subsistenz(t) + Ci_nuss_genuss(t) + Ci_nuss_invest(t)      # Gesamte Kokosnussverwendung (Konsum und Investition)
Randbedingungen: Ci_fisch(t) - Ci_fisch_invest(t) >= 0, Ci_nuss(t) - Ci_nuss_invest(t) >= 0

Konsumaufteilung in Subsistenz, Genuss und Investitionen:
Ci_fisch_subsistenz(t) = wi_fisch * (Ci_fisch(t) - Ci_fisch_invest(t))   # Subsistenz-Anteil am Fischkonsum (wi_fisch wird optimiert)
Ci_fisch_genuss(t) = (1 - wi_fisch) * (Ci_fisch(t) - Ci_fisch_invest(t)) # Genuss-Anteil am Fischkonsum
Ci_nuss_subsistenz(t) = wi_nuss * (Ci_nuss(t) - Ci_nuss_invest(t))       # Subsistenz-Anteil am Nusskonsum (wi_nuss wird optimiert)
Ci_nuss_genuss(t) = (1 - wi_nuss) * (Ci_nuss(t) - Ci_nuss_invest(t))     # Genuss-Anteil am Nusskonsum
Randbedingungen: 0 <= wi_fisch, wi_nuss <= 1

Gesamtproduktion = Gesamtgüterverwendung:
C1_fisch(t) + C2_fisch(t) = Y1_fisch(t) + Y2_fisch(t) # Kontrolle: Fischverwendung = Fischproduktion
C1_nuss(t) + C2_nuss(t) = Y1_nuss(t) + Y2_nuss(t)     # Kontrolle: Kokosnussverwendung = Kokosnussproduktion

Tauschhandel:
C1_fisch(t) = Y1_fisch(t) - tau1_fisch * Y1_fisch(t)
C2_fisch(t) = Y2_fisch(t) + tau1_fisch * Y1_fisch(t)
C1_nuss(t) = Y1_nuss(t) + tau2_nuss * Y2_nuss(t)
C2_nuss(t) = Y2_nuss(t) - tau2_nuss * Y2_nuss(t)

Tauschpreis in Nüssen pro Fisch: (tau2_nuss * Y2_nuss(t)) / (tau1_fisch * Y1_fisch(t))

Biologische Restriktion (ein Mensch benötige mind. Li_subsistenz_min Stunden Schlaf, Ci_fisch_subsistenz_min Fische und Ci_nuss_subsistenz_min Kokosnüsse zum Überleben):
Li_subsistenz >= Li_subsistenz_min                # z.B. Li_subsistenz_min = 8 Stunden Schlafen, Essen, Kochen
Ci_fisch_subsistenz(t) >= Ci_fisch_subsistenz_min # z.B. Ci_fisch_subsistenz_min = 2 Fische
Ci_nuss_subsistenz(t) >= Ci_nuss_subsistenz_min   # z.B. Ci_nuss_subsistenz_min = 4 Kokosnüsse

Nutzenfunktionen:  
Ui_subsistenz(Li_subsistenz, Ci_fisch_subsistenz(t), Ci_nuss_subsistenz(t)) = Ai_subsistenz * Li_subsistenz^alphai_subsistenz * Ci_fisch_subsistenz(t)^betai_subsistenz * Ci_nuss_subsistenz(t)^gammai_subsistenz 
Ui_genuss(Li_genuss(t), Ci_fisch_genuss(t), Ci_nuss_genuss(t)) = Ai_genuss * (Li_genuss(t) + Ci_fisch_genuss(t)/ri_fisch_genuss + Ci_nuss_genuss(t)/ri_nuss_genuss)^alphai_genuss
Ui(Li_subsistenz, Ci_fisch_subsistenz(t), Ci_nuss_subsistenz(t), Li_genuss(t), Ci_fisch_genuss(t), Ci_nuss_genuss(t)) = Ui_subsistenz(Li_subsistenz, Ci_fisch_subsistenz(t), Ci_nuss_subsistenz(t)) + Ui_genuss(Li_genuss(t), Ci_fisch_genuss(t), Ci_nuss_genuss(t))

Autarkienutzen (tau1_fisch = tau2_nuss = 0):
Li_genuss_autarkie(t) = L_max - Li_fisch_autarkie - Li_nuss_autarkie - Li_subsistenz_autarkie - Li_fisch_invest_autarkie(t) - Li_nuss_invest_autarkie(t) - Li_fisch_research - Li_nuss_research, Li_genuss_autarkie(t) >= 0
Yi_fisch_autarkie(t) = Ai_fisch(Ri_fisch_autarkie(t)) * Li_fisch_autarkie^alphai_L_fisch * (Ki_fisch_basis + Ki_fisch_autarkie(t))^alphai_K_fisch
Yi_nuss_autarkie(t) = Ai_nuss(Ri_nuss_autarkie(t)) * Li_nuss_autarkie^alphai_L_nuss * (Ki_nuss_basis + Ki_nuss_autarkie(t))^alphai_K_nuss
Ki_fisch_autarkie(t) = K_fisch_max * (1 - exp(-lambdai_fisch_invest_autarkie * t))
Li_fisch_invest_autarkie(t) = (((K_fisch_max * lambdai_fisch_invest_autarkie*exp(-lambdai_fisch_invest_autarkie*t))/Ai_fisch_invest)^(1/alphai_fisch_invest))/ri_fisch_invest
Ki_nuss_autarkie(t) = K_nuss_max * (1 - exp(-lambdai_nuss_invest_autarkie * t))
Li_nuss_invest_autarkie(t) = (((K_nuss_max * lambdai_nuss_invest_autarkie*exp(-lambdai_nuss_invest_autarkie*t))/Ai_nuss_invest)^(1/alphai_nuss_invest))/ri_nuss_invest
Ri_fisch_autarkie(t) = (1 + (1-alphai_research) * Ai_fisch_research * Li_fisch_research_autarkie * t)^(1/(1-alphai_research)) - 1 für 0 <= alphai_research < 1
Ri_fisch_autarkie(t) = (Ai_fisch_research + 1)^t - 1 für alphai_research = 1
Ri_nuss_autarkie(t) = (1 + (1-alphai_research) * Ai_nuss_research * Li_nuss_research_autarkie * t)^(1/(1-alphai_research)) - 1 für 0 <= alphai_research < 1
Ri_nuss_autarkie(t) = (Ai_nuss_research + 1)^t - 1 für alphai_research = 1
Ci_fisch_invest_autarkie(t) = ri_fisch_invest * Li_fisch_invest_autarkie(t)
Ci_nuss_invest_autarkie(t) = ri_nuss_invest * Li_nuss_invest_autarkie(t)
Ci_fisch_subsistenz_autarkie(t) = wi_fisch_autarkie * (Yi_fisch_autarkie(t) - Ci_fisch_invest_autarkie(t))
Ci_fisch_genuss_autarkie(t) = (1 - wi_fisch_autarkie) * (Yi_fisch_autarkie(t) - Ci_fisch_invest_autarkie(t))
Ci_nuss_subsistenz_autarkie(t) = wi_nuss_autarkie * (Yi_nuss_autarkie(t) - Ci_nuss_invest_autarkie(t))
Ci_nuss_genuss_autarkie(t) = (1 - wi_nuss_autarkie) * (Yi_nuss_autarkie(t) - Ci_nuss_invest_autarkie(t))
Ui_autarkie(t) = Ui(Li_subsistenz_autarkie, Ci_fisch_subsistenz_autarkie(t), Ci_nuss_subsistenz_autarkie(t), Li_genuss_autarkie(t), Ci_fisch_genuss_autarkie(t), Ci_nuss_genuss_autarkie(t))
Maximierung der diskontierten Autarkienutzen liefert die Entscheidungsvariablen und Investitionen unter Autarkiebedingungen:
Vi_autarkie = sum_{0<=t<t_max} (δ_i^t * Ui_autarkie(t)) → max

Nashlösung: 
Vi = sum_{0<=t<t_max} (δ_i^t * Ui(t)) # Diskontierter Nutzen
Vi_autarkie = sum_{0<=t<t_max} (δ_i^t * Ui_autarkie(t)) # Diskontierter Autarkienutzen
log(V1 - V1_autarkie) + log(V2 - V2_autarkie) → max mit den Randbedingungen V1 - V1_autarkie >= ε und V2 - V2_autarkie >= ε

Erläuterungen siehe auch MiniWM_Teil02_RobinsonFreitag.md.
"""
import datetime
from enum import IntEnum
import itertools
import json
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
import scipy # pip install scipy
import scipy.optimize
import sympy as sp # pip install sympy
import time
from typing import Literal, Union
from mini_wm_robinson_two_persons_test import axes_search, search_Ui_autarkie_diskont, test_maximize_nash_product

# Beispielparameter (Robinson: i=1, Freitag: i=2)

# Produktion
alphai_L_fisch = { 1: 0.5, 2: 0.5 }        # Robinsons und Freitags Arbeitselastizität in der Fischproduktion
alphai_L_nuss = { 1: 0.4, 2: 0.4 }         # Robinsons und Freitags Arbeitselastizität in der Kokosnussproduktion
alphai_K_fisch = { 1: 1.0, 2: 1.0 }        # Robinsons und Freitags Kapitalelastizität in der Fischproduktion
alphai_K_nuss = { 1: 1.0, 2: 1.0 }         # Robinsons und Freitags Kapitalelastizität in der Kokosnussproduktion
Ai_fisch = { 1: 6, 2: 3 }                  # Robinsons und Freitags Basis-Produktivität der Fischproduktion (exkl. Forschung)
Ai_nuss = { 1: 4, 2: 8 }                   # Robinsons und Freitags Basis-Produktivität der Kokosnussproduktion (exkl. Forschung)

# Subsistenzkonsum
Ai_subsistenz = { 1: 1, 2: 1 }             # Niveauparameter von Robinsons und Freitags Subsistenzkonsum
alphai_subsistenz = { 1: 1, 2: 1 }         # Elastizitäten von Robinsons und Freitags Zeiteinsätzen im Subsistenzkonsum
betai_subsistenz = { 1: 0.1, 2: 0.1 }      # Elastizitäten von Robinsons und Freitags Fischeinsätzen im Subsistenzkonsum
gammai_subsistenz = { 1: 0.1, 2: 0.1 }     # Elastizitäten von Robinsons und Freitags Nusseinsätzen im Subsistenzkonsum

# Genusskonsum
Ai_genuss = { 1: 1, 2: 1 }                 # Niveauparameter von Robinsons und Freitags Genusskonsum
alphai_genuss = { 1: 0.5, 2: 0.5 }         # Elastizitäten von Robinsons und Freitags Einsätzen im Genusskonsum
ri_fisch_genuss = { 1: 3, 2: 3 }           # Der Nutzen aus 1 Stunde Erholung entspricht dem Nutzen aus Genuss von ri_fisch_genuss[i] = 3 Fischen
ri_nuss_genuss = { 1: 5, 2: 5 }            # Der Nutzen aus 1 Stunde Erholung entspricht dem Nutzen aus Genuss von ri_nuss_genuss[i] = 5 Nüssen

# Kapital
Ki_fisch_basis = { 1: 1, 2: 1 }            # Robinsons und Freitags Anfangskapital in der Fischproduktion
Ki_nuss_basis = { 1: 1, 2: 1 }             # Robinsons und Freitags Anfangskapital in der Kokosnussproduktion
K_fisch_max = 100                          # Grenze des zusätzlichen von Robinson und Freitag durch Investitionen akkumulierten Fisch-Kapitals (begrenzte Fischzuchtfläche)
K_nuss_max = 100                           # Grenze des zusätzlichen von Robinson und Freitag durch Investitionen akkumulierten Kokosnuss-Kapitals (begrenzte Anbaufläche)

# Investition
alphai_fisch_invest = { 1: 1.0, 2: 1.0 }   # Elastizität der Investitionen in Fischzucht
alphai_nuss_invest = { 1: 1.0, 2: 1.0 }    # Elastizität der Investitionen in Kokosnussanbau
Ai_fisch_invest = { 1: 2, 2: 2 }           # Produktivität der Investitionen in Fischzucht
Ai_nuss_invest = { 1: 2, 2: 2 }            # Produktivität der Investitionen in Kokosnussanbau
ri_fisch_invest = { 1: 5, 2: 5 }           # In 1 Zeiteinheit werden ri_fisch_invest[i] Fische in der Fischzucht investiert
ri_nuss_invest = { 1: 10, 2: 10 }          # In 1 Zeiteinheit werden ri_nuss_invest[i] Kokosnüsse in den Kokosnussanbau investiert

# Forschung
Ai_fisch_research = { 1: 0.3, 2: 0.1 }     # Robinsons und Freitags Produktivität der Forschung in Fischtechnologie
Ai_nuss_research = { 1: 0.1, 2: 0.3 }      # Robinsons und Freitags Produktivität der Forschung in Kokosnusstechnologie
alphai_research = 0.5                      # Robinsons und Freitags Arbeitselastizität in der Forschung: alphai_research entscheiden über das Wachstum von Ri_fisch(t), Ri_nuss(t). Mit alphai_research << 0 wächst die Produktivität extrem langsam, mit alphai_research = 0 linear, mit alphai_research = 1 exponentiell, und mit 0 < alphai_research < 1 zwischen linear und exponentiell.  
gammai_fisch_research = { 1: 0.1, 2: 0.1 } # Robinsons und Freitags Forschungsproduktivität in der Fischproduktion
gammai_nuss_research = { 1: 0.1, 2: 0.1 }  # Robinsons und Freitags Forschungsproduktivität in der Kokosnussproduktion

# Zeitrestriktion
L_max = 24                                 # Max. Arbeitseinsatz pro Person: 24 Stunden
t_max = 101                                # Maximale Anzahl von Zeiteinheiten, um die intertemporale Optimierung durchzuführen

# Biologische Restriktion (Mindestbedarf zur Subsistenz):
Li_subsistenz_min = 8                      # Li_subsistenz >= Li_subsistenz_min, mind. Li_subsistenz_min Stunden zur Subsistenz (Schlafen, Kochen und Essen)
Ci_fisch_subsistenz_min = 1                # Ci_fisch_subsistenz(t) >= Ci_fisch_subsistenz_min, mind. Ci_fisch_subsistenz_min Fische zur Subsistenz
Ci_nuss_subsistenz_min = 1                 # Ci_nuss_subsistenz(t) >= Ci_nuss_subsistenz_min, mind. Ci_nuss_subsistenz_min Kokosnüsse zur Subsistenz

# Plots
plt_cnt = 0 # Globaler Zähler für Diagramme und png-Dateien
plt_now_str = f"{datetime.datetime.now():%Y%m%d_%H%M%S}" # Globaler Zeitstempel für Diagramme und png-Dateien

# Freie Entscheidungsvariablen sind 20 Unbekannte (Robinson: i=1, Freitag: i=2):
# Li_fisch (Arbeitseinsätze für Fischproduktion), Li_nuss (Arbeitseinsätze für Kokosnussproduktion), Li_subsistenz (Zeiten für Subsistenz), Li_fisch_research (Arbeitseinsätze für Forschung in Fischtechnologie), Li_nuss_research (Arbeitseinsätze für Forschung in Kokosnusstechnologie)
# lambdai_fisch_invest (Wachstum Fischzucht-Kapital), lambdai_nuss_invest (Wachstum Kokosnuss-Kapital), 
# wi_fisch (Subsistenz-Anteil am Fischkonsum), wi_nuss (Subsistenz-Anteil am Nusskonsum),
# tau1_fisch (Robinsons Tauschanteil), tau2_nuss (Freitags Tauschanteil)
# Enumeration der Indizes der Entscheidungsvariablen im zu optimierenden Parametervektor x:
class X0IDX(IntEnum):        # Indizes der Entscheidungsvariablen für Autarkienutzen
    L_fisch = 0              # Li_fisch_autarkie = x[X0IDX.L_fisch]: Autarkie-Arbeitseinsatz für Fischproduktion
    L_nuss = 1               # Li_nuss_autarkie = x[X0IDX.L_nuss]: Autarkie-Arbeitseinsatz für Kokosnussproduktion
    L_subsistenz = 2         # Li_subsistenz_autarkie = x[X0IDX.L_subsistenz]: Autarkie-Zeit für Subsistenz
    L_fisch_research = 3     # Li_fisch_research_autarkie = x[X0IDX.L_fisch_research]: Autarkie-Arbeitseinsatz für Forschung in Fischtechnologie
    L_nuss_research = 4      # Li_nuss_research_autarkie = x[X0IDX.L_nuss_research]: Autarkie-Arbeitseinsatz für Forschung in Kokosnusstechnologie
    lambda_fisch_invest = 5  # lambdai_fisch_invest_autarkie = x[X0IDX.lambda_fisch_invest]: Autarkie-Wachstum für Fischzucht-Kapital 
    lambda_nuss_invest = 6   # lambdai_nuss_invest_autarkie = x[X0IDX.lambda_nuss_invest]: Autarkie-Wachstum für Kokosnuss-Kapital
    w_fisch = 7              # wi_fisch_autarkie = x[X0IDX.w_fisch]: Autarkie-Subsistenz-Anteil am Fischkonsum
    w_nuss = 8               # wi_nuss_autarkie = x[X0IDX.w_nuss]: Autarkie-Subsistenz-Anteil am Nusskonsum
class X1IDX(IntEnum):        # Indizes der Entscheidungsvariablen für Robinson
    L_fisch = 0              # L1_fisch = x[X1IDX.L_fisch]: Robinsons Arbeitseinsatz für Fischproduktion
    L_nuss = 1               # L1_nuss = x[X1IDX.L_nuss]: Robinsons Arbeitseinsatz für Kokosnussproduktion
    L_subsistenz = 2         # L1_subsistenz = x[X1IDX.L_subsistenz]: Robinsons Zeit für Subsistenz
    L_fisch_research = 3     # L1_fisch_research = x[X1IDX.L_fisch_research]: Robinsons Arbeitseinsatz für Forschung in Fischtechnologie
    L_nuss_research = 4      # L1_nuss_research = x[X1IDX.L_nuss_research]: Robinsons Arbeitseinsatz für Forschung in Kokosnusstechnologie
    lambda_fisch_invest = 5  # lambda1_fisch_invest = x[X1IDX.lambda_fisch_invest]: Wachstum für Robinsons Fischzucht-Kapital 
    lambda_nuss_invest = 6   # lambda1_nuss_invest = x[X1IDX.lambda_nuss_invest]: Wachstum für Robinsons Kokosnuss-Kapital
    w_fisch = 7              # w1_fisch = x[X1IDX.w_fisch]: Robinsons Subsistenz-Anteil am Fischkonsum
    w_nuss = 8               # w1_nuss = x[X1IDX.w_nuss]: Robinsons Subsistenz-Anteil am Nusskonsum
    tau = 9                  # tau1_fisch = x[X1IDX.tau]: Robinsons Tauschanteil (Robinson tauscht tau1_fisch*Y1_fisch seiner Fische gegen Kokosnüsse)
class X2IDX(IntEnum):        # Indizes der Entscheidungsvariablen für Freitag
    L_fisch = 10             # L2_fisch = x[X2IDX.L_fisch]: Freitags Arbeitseinsatz für Fischproduktion
    L_nuss = 11              # L2_nuss = x[X2IDX.L_nuss]: Freitags Arbeitseinsatz für Kokosnussproduktion
    L_subsistenz = 12        # L2_subsistenz = x[X2IDX.L_subsistenz]: Freitags Zeit für Subsistenz
    L_fisch_research = 13    # L2_fisch_research = x[X2IDX.L_fisch_research]: Freitags Arbeitseinsatz für Forschung in Fischtechnologie
    L_nuss_research = 14     # L2_nuss_research = x[X2IDX.L_nuss_research]: Freitags Arbeitseinsatz für Forschung in Kokosnusstechnologie
    lambda_fisch_invest = 15 # lambda2_fisch_invest = x[X2IDX.lambda_fisch_invest]: Wachstum für Freitags Fischzucht-Kapital 
    lambda_nuss_invest = 16  # lambda2_nuss_invest = x[X2IDX.lambda_nuss_invest]: Wachstum für Freitags Kokosnuss-Kapital
    w_fisch = 17             # w2_fisch = x[X2IDX.w_fisch]: Freitags Subsistenz-Anteil am Fischkonsum
    w_nuss = 18              # w2_nuss = x[X2IDX.w_nuss]: Freitags Subsistenz-Anteil am Nusskonsum
    tau = 19                 # tau2_nuss = x[X2IDX.tau]: Freitags Tauschanteil (Freitag tauscht tau2_nuss*Y2_nuss seiner Kokosnüsse gegen Fische)
xidx = (X0IDX, X1IDX, X2IDX) # Indizes der Entscheidungsvariablen: xidx[0] für Autarkie, xidx[1] für Robinson, xidx[2] für Freitag

# Indizes der Randbedingungen für Constraints.update:
class CONSTRAINTIDX(IntEnum):
    Ci_fisch_subsistenz = 0 # Biologische Restriktion (Mindestbedarf zur Subsistenz): Ci_fisch_subsistenz(t) >= Ci_fisch_subsistenz_min <=> (Ci_fisch_subsistenz(t)/Ci_fisch_subsistenz_min)-1 >= 0
    Ci_nuss_subsistenz = 1  # Biologische Restriktion (Mindestbedarf zur Subsistenz): Ci_nuss_subsistenz(t) >= Ci_nuss_subsistenz_min <=> (Ci_nuss_subsistenz(t)/Ci_nuss_subsistenz_min)-1 >= 0
    Li_genuss = 2           # Randbedingung: Li_genuss(t) >= 0 <=> Li_genuss(t) / L_max >= 0
    Vi = 3                  # Vi - Vi_autarkie >= epsilon <=> (Vi-Vi_autarkie-epsilon)/max(1,|Vi_autarkie|) >= 0

# Randbedingungen: Falls constraints.min() < min_constraints, sind Randbedingungen verletzt.
class Constraints:
    def __init__(self, max_constraint = np.inf):
        self.constraints_arr = [[max_constraint] * 4, [max_constraint] * 4]
    def update(self, i: Literal[1, 2], idx: CONSTRAINTIDX, value: float):
        self.constraints_arr[i-1][idx] = min(value, self.constraints_arr[i-1][idx])
    def array(self):
        return np.concatenate((self.constraints_arr[0], self.constraints_arr[1]))
    def autarkie_array(self, i):
        return self.constraints_arr[i-1][0:-1] # Autarkie ohne Vi constraints
    def min(self):
        return min(min(self.constraints_arr[0]), min(self.constraints_arr[1]))

# Settings für maximize_nash_product: Diskontfaktoren delta_1, delta_2 und Startwerte x0 für die Optimierung des Nashproduktes
class Settings:
    def __init__(self, delta_1: float, delta_2: float, x0_autarkie_1: np.ndarray, x0_autarkie_2: np.ndarray, x0: np.ndarray = None, plot_log: bool = True):
        self.delta_1 = delta_1 # Robinsons Diskontfaktor
        self.delta_2 = delta_2 # Freitags Diskontfaktor
        self.x0_autarkie_1 = np.copy(x0_autarkie_1) # Robinsons Startwert x0_autarkie = [ Li_fisch, Li_nuss, Li_subsistenz, Li_fisch_research, Li_nuss_research, lambdai_fisch_invest, lambdai_nuss_invest, wi_fisch, wi_nuss ]
        self.x0_autarkie_2 = np.copy(x0_autarkie_2) # Freitags Startwert x0_autarkie = [ Li_fisch, Li_nuss, Li_subsistenz, Li_fisch_research, Li_nuss_research, lambdai_fisch_invest, lambdai_nuss_invest, wi_fisch, wi_nuss ]
        self.x0 = x0 # Startwert für maximize_nash_product_x0, default: None für Startwert x0 = [x1_autarkie, 0.1, x2_autarkie, 0.1]
        self.plot_log = plot_log # True: Log. plot der Ergebnisse, andernfall linearer plot

# Kapitalwerte K1, K2, R1, R2 (Akkumulation über der Zeit)
class Capitals:
    def __init__(self, K1_fisch: float = 0, K1_nuss: float = 0, R1_fisch: float = 0, R1_nuss: float = 0, K2_fisch: float = 0, K2_nuss: float = 0, R2_fisch: float = 0, R2_nuss: float = 0):
        self.Ki_fisch = {1: K1_fisch, 2: K2_fisch}
        self.Ki_nuss = {1: K1_nuss, 2: K2_nuss}
        self.Ri_fisch = {1: R1_fisch, 2: R2_fisch}
        self.Ri_nuss = {1: R1_nuss, 2: R2_nuss}
    def copy(self):
        return Capitals(self.Ki_fisch[1], self.Ki_nuss[1], self.Ri_fisch[1], self.Ri_nuss[1], self.Ki_fisch[2], self.Ki_nuss[2], self.Ri_fisch[2], self.Ri_nuss[2])
    def __str__(self):
        return f"[{self.Ki_fisch[1]:.2f},{self.Ki_nuss[1]:.2f},{self.Ri_fisch[1]:.2f},{self.Ri_nuss[1]:.2f},{self.Ki_fisch[2]:.2f},{self.Ki_nuss[2]:.2f},{self.Ri_fisch[2]:.2f},{self.Ri_nuss[2]:.2f}]"

# Ergebnisse der Maximierung des Nashproduktes mit maximize_nash_product()
class MaxNPResult:
    def __init__(self, x: np.ndarray, x1_autarkie: np.ndarray, V1_autarkie: float, x2_autarkie: np.ndarray, V2_autarkie: float, U1: float, U2: float, logNP: float, constraints_min: float, success: bool, usable: bool):
        self.x = np.copy(x)
        self.x1_autarkie = np.copy(x1_autarkie)
        self.V1_autarkie = V1_autarkie
        self.x2_autarkie = np.copy(x2_autarkie)
        self.V2_autarkie = V2_autarkie
        self.U1 = U1
        self.U2 = U2
        self.logNP = logNP
        self.constraints_min = constraints_min
        self.success = success
        self.usable = usable

# Kennwerte (Produktion Yi_fisch, Yi_nuss, Kapitalwerte Ki_fisch, Ki_nuss, Ri_fisch, Ri_nuss und Nutzen U1, U2) über der Zeit t
class KeyFigures:
    def __init__(self, delta_1: float, delta_2: float):
        self.deltai = { 1: delta_1, 2: delta_2 }
        self.t_values = np.zeros(0, dtype=float)
        self.Yi_fisch_values = { 1: np.zeros(0, dtype=float), 2: np.zeros(0, dtype=float) }
        self.Yi_nuss_values = { 1: np.zeros(0, dtype=float), 2: np.zeros(0, dtype=float) }
        self.Ki_fisch_values = { 1: np.zeros(0, dtype=float), 2: np.zeros(0, dtype=float) }
        self.Ki_nuss_values = { 1: np.zeros(0, dtype=float), 2: np.zeros(0, dtype=float) }
        self.Ri_fisch_values = { 1: np.zeros(0, dtype=float), 2: np.zeros(0, dtype=float) }
        self.Ri_nuss_values = { 1: np.zeros(0, dtype=float), 2: np.zeros(0, dtype=float) }
        self.Ui_values = { 1: np.zeros(0, dtype=float), 2: np.zeros(0, dtype=float) }
        self.tau_nuts_per_fish = np.zeros(0, dtype=float)
    def to_json(self):
        return {"delta1": self.deltai[1], "delta2": self.deltai[2], "t": self.t_values.tolist(), 
                "Y1_fisch": self.Yi_fisch_values[1].tolist(), "Y2_fisch": self.Yi_fisch_values[2].tolist(), "Y1_nuss": self.Yi_nuss_values[1].tolist(), "Y2_nuss": self.Yi_nuss_values[2].tolist(), 
                "K1_fisch": self.Ki_fisch_values[1].tolist(), "K2_fisch": self.Ki_fisch_values[2].tolist(), "K1_nuss": self.Ki_nuss_values[1].tolist(), "K2_nuss": self.Ki_nuss_values[2].tolist(), 
                "R1_fisch": self.Ri_fisch_values[1].tolist(), "R2_fisch": self.Ri_fisch_values[2].tolist(), "R1_nuss": self.Ri_nuss_values[1].tolist(), "R2_nuss": self.Ri_nuss_values[2].tolist(), 
                "U1": self.Ui_values[1].tolist(), "U2": self.Ui_values[2].tolist(), "tau_nuts_per_fish": self.tau_nuts_per_fish.tolist()}

# Shortcut zur Formatierung aller Elemente einer Liste
def fmt(list, fmt_str = "{:5.2f}", sep = ",", pre = "[", post="]"):
    return pre + sep.join([fmt_str.format(_elem) for _elem in list]) + post

# Nicht-rekursive Approximation des Wissenkapitals für α_research = 1 und α_research < 1
# R(0) = R0
# R(t) = ((1 + R0)^(1-α_research) + (1-α_research) * A_research * L_research * t)^(1/(1-α_research)) - 1 für 0 <= α_research < 1
# R(t) = (1 + R0) * exp(A_research * L_research * t) für α_research = 1
def calc_research(t, A_research, L_research, alpha_research, R0):
    if t < 1e-8 or L_research < 1e-8:
        return R0
    if abs(alpha_research - 1.0) < 1e-8:
        return np.expm1(np.log1p(R0) + A_research * L_research * t)
    one_minus_alpha = 1.0 - alpha_research
    Rt_base = (1.0 + R0)**one_minus_alpha + one_minus_alpha * A_research * L_research * t
    Rt = Rt_base**(1.0 / one_minus_alpha) - 1.0
    return Rt

# Implementierung des Modells. Argumente der calc_-Funktionen sind: 
# i=1: Robinson, i=2: Freitag, 
# t: Zeitschritt, 
# x: Vektor der Entscheidungsvariablen mit Indizes xidx[i], 
# K_start: Startkapital, 
# constraints: constraint-Verletzungen (constraints_arr < 0: Randbedingung verletzt)

# Berechnung des Wissenkapitals in der Fischtechnologie: 
# Ri_fisch(t) = (1 + (1-alphai_research) * Ai_fisch_research * Li_fisch_research * t)^(1/(1-alphai_research)) - 1 für 0 <= alphai_research < 1
# Ri_fisch(t) = (Ai_fisch_research * Li_fisch_research + 1)^t - 1 für alphai_research = 1
def calc_Ri_fisch(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    return calc_research(t, Ai_fisch_research[i], x[xidx[i].L_fisch_research], alphai_research, K_start.Ri_fisch[i])

# Berechnung des Wissenkapitals in der Kokosnusstechnologie: 
# Ri_nuss(t) = (1 + (1-alphai_research) * Ai_nuss_research * Li_nuss_research * t)^(1/(1-alphai_research)) - 1 für 0 <= alphai_research < 1
# Ri_nuss(t) = (Ai_nuss_research * Li_nuss_research + 1)^t - 1 für alphai_research = 1
def calc_Ri_nuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    return calc_research(t, Ai_nuss_research[i], x[xidx[i].L_nuss_research], alphai_research, K_start.Ri_nuss[i])

# Berechnung des Fisch-Produktivität: Ai_fisch(Ri_fisch(t)) = Ai_fisch_0 + gammai_fisch_research * Ri_fisch(t)
def calc_Ai_fisch(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    return Ai_fisch[i] + gammai_fisch_research[i] * calc_Ri_fisch(i, t, x, K_start, constraints)

# Berechnung des Kokosnuss-Produktivität: Ai_nuss(Ri_nuss(t)) = Ai_nuss_0 + gammai_nuss_research * Ri_nuss(t)
def calc_Ai_nuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    return Ai_nuss[i] + gammai_nuss_research[i] * calc_Ri_nuss(i, t, x, K_start, constraints)

# Berechnung des Fischzucht-Kapitals: Ki_fisch(t) = K_fisch_max * (1 - exp(-lambdai_fisch_invest * t))
def calc_Ki_fisch(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    lambdai_fisch_invest = x[xidx[i].lambda_fisch_invest]
    return (K_fisch_max - K_start.Ki_fisch[i]) * (1 - np.exp(-lambdai_fisch_invest * t)) + K_start.Ki_fisch[i]

# Berechnung des Kokosnuss-Kapitals: Ki_nuss(t) = K_nuss_max * (1 - exp(-lambdai_nuss_invest * t))
def calc_Ki_nuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    lambdai_nuss_invest = x[xidx[i].lambda_nuss_invest]
    return (K_nuss_max - K_start.Ki_nuss[i]) * (1 - np.exp(-lambdai_nuss_invest * t)) + K_start.Ki_nuss[i]

# Berechung des Arbeitseinsatzes, der in die Fischzucht investiert wird: Li_fisch_invest(t) = Ci_fisch_invest(t) / ri_fisch_invest, d.h. in Li_fisch_invest Zeiteinheiten werden Ci_fisch_invest Fische investiert.
def calc_Li_fisch_invest(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    return calc_Ci_fisch_invest(i, t, x, K_start, constraints) / ri_fisch_invest[i]

# Berechung des Arbeitseinsatzes, der in den Kokosnussanbau investiert wird: Li_nuss_invest(t) = Ci_nuss_invest(t) / ri_nuss_invest, d.h. in Li_nuss_invest Zeiteinheiten werden Ci_nuss_invest Kokosnüsse investiert.
def calc_Li_nuss_invest(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    return calc_Ci_nuss_invest(i, t, x, K_start, constraints) / ri_nuss_invest[i]

# Berechung der Anzahl Fische, die in die Fischzucht investiert werden:
# Investition in Fischzucht: Ii_fisch(t) = Ai_fisch_invest * Ci_fisch_invest(t)^alphai_fisch_invest = dKi_fisch(t)/dt = (K_fisch_max - K_fisch_start) * lambdai_fisch_invest * exp(-lambdai_fisch_invest * t)
def calc_Ci_fisch_invest(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    lambdai_fisch_invest = x[xidx[i].lambda_fisch_invest]
    K_fisch_remaining = max(0.0, K_fisch_max - K_start.Ki_fisch[i])
    Ii_fisch = K_fisch_remaining * lambdai_fisch_invest * np.exp(-lambdai_fisch_invest * t) # Investition Ii_fisch(t) =  dKi_fisch(t)/dt
    Ci_fisch_invest = (Ii_fisch / Ai_fisch_invest[i])**(1/alphai_fisch_invest[i])           # Investition Ii_fisch(t) = Ai_fisch_invest * Ci_fisch_invest(t)^alphai_fisch_invest
    return Ci_fisch_invest

# Berechung der Anzahl Kokosnüsse, die in den Kokosnussanbau investiert werden:
# Investition in Kokosnussanbau: Ii_nuss(t) = Ai_nuss_invest * Ci_nuss_invest(t)^alphai_nuss_invest = dKi_nuss(t)/dt = (K_nuss_max - K_nuss_start) * lambdai_nuss_invest * exp(-lambdai_nuss_invest * t)
def calc_Ci_nuss_invest(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    lambdai_nuss_invest = x[xidx[i].lambda_nuss_invest]
    K_nuss_remaining = max(0.0, K_nuss_max - K_start.Ki_nuss[i])
    Ii_nuss = K_nuss_remaining * lambdai_nuss_invest * np.exp(-lambdai_nuss_invest * t) # Investition Ii_nuss(t) =  dKi_nuss(t)/dt
    Ci_nuss_invest = (Ii_nuss / Ai_nuss_invest[i])**(1/alphai_nuss_invest[i])           # Investition Ii_nuss(t) = Ai_nuss_invest * Ci_nuss_invest(t)^alphai_nuss_invest
    return Ci_nuss_invest

# Berechung der Fischproduktion mit Anfangskapital Ki_fisch_basis: Yi_fisch(t, Li_fisch) = Ai_fisch(Ri_fisch(t)) * Li_fisch^alphai_L_fisch * (Ki_fisch_basis + Ki_fisch(t))^alphai_K_fisch
def calc_Yi_fisch(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Ki_fisch = calc_Ki_fisch(i, t, x, K_start, constraints)
    return calc_Ai_fisch(i, t, x, K_start, constraints) * ((x[xidx[i].L_fisch])**alphai_L_fisch[i]) * ((Ki_fisch_basis[i] + Ki_fisch)**alphai_K_fisch[i])

# Berechung der Kokosnussproduktion mit Anfangskapital Ki_nuss_basis: Yi_nuss(t, Li_nuss) = Ai_nuss(Ri_nuss(t)) * Li_nuss^alphai_L_nuss * (Ki_nuss_basis + Ki_nuss(t))^alphai_K_nuss
def calc_Yi_nuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Ki_nuss = calc_Ki_nuss(i, t, x, K_start, constraints)
    return calc_Ai_nuss(i, t, x, K_start, constraints) * ((x[xidx[i].L_nuss])**alphai_L_nuss[i]) * ((Ki_nuss_basis[i] + Ki_nuss)**alphai_K_nuss[i])

# Berechung des Subsistenz-Anteils am Fischkonsum: Ci_fisch_subsistenz(t) = wi_fisch * (Ci_fisch(t) - Ci_fisch_invest(t))
def calc_Ci_fisch_subsistenz(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Ci_fisch = calc_Ci_fisch(i, t, x, K_start, constraints)
    Ci_fisch_invest = calc_Ci_fisch_invest(i, t, x, K_start, constraints)
    Ci_fisch_subsistenz = x[xidx[i].w_fisch] * (Ci_fisch - Ci_fisch_invest)
    # Biologische Restriktion (Mindestbedarf zur Subsistenz): Ci_fisch_subsistenz(t) >= Ci_fisch_subsistenz_min, dadurch auch Randbedingung: Ci_fisch(t) - Ci_fisch_invest(t) >= 0
    # Wertenormierung: Ci_fisch_subsistenz(t) >= Ci_fisch_subsistenz_min <=> (Ci_fisch_subsistenz(t)/Ci_fisch_subsistenz_min)-1 >= 0
    constraints.update(i, CONSTRAINTIDX.Ci_fisch_subsistenz, Ci_fisch_subsistenz / Ci_fisch_subsistenz_min - 1) 
    return Ci_fisch_subsistenz

# Berechung des Subsistenz-Anteils am Nusskonsum: Ci_nuss_subsistenz(t) = wi_nuss * (Ci_nuss(t) - Ci_nuss_invest(t))
def calc_Ci_nuss_subsistenz(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Ci_nuss = calc_Ci_nuss(i, t, x, K_start, constraints)
    Ci_nuss_invest = calc_Ci_nuss_invest(i, t, x, K_start, constraints)
    Ci_nuss_subsistenz = x[xidx[i].w_nuss] * (Ci_nuss - Ci_nuss_invest)
    # Biologische Restriktion (Mindestbedarf zur Subsistenz): Ci_nuss_subsistenz(t) >= Ci_nuss_subsistenz_min, dadurch auch Randbedingung: Ci_nuss(t) - Ci_nuss_invest(t) >= 0
    # Wertenormierung: Ci_nuss_subsistenz(t) >= Ci_nuss_subsistenz_min <=> (Ci_nuss_subsistenz(t)/Ci_nuss_subsistenz_min)-1 >= 0
    constraints.update(i, CONSTRAINTIDX.Ci_nuss_subsistenz, Ci_nuss_subsistenz / Ci_nuss_subsistenz_min - 1) 
    return Ci_nuss_subsistenz

# Berechnung der Zeit für Genusskonsum, Zeitrestriktion: Li_genuss(t) = L_max - Li_fisch - Li_nuss - Li_subsistenz - Li_fisch_invest(t) - Li_nuss_invest(t) - Li_fisch_research - Li_nuss_research, L_max = 24 Stunden
def calc_Li_genuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Li_genuss = L_max - x[xidx[i].L_fisch] - x[xidx[i].L_nuss] - x[xidx[i].L_subsistenz] - calc_Li_fisch_invest(i, t, x, K_start, constraints) - calc_Li_nuss_invest(i, t, x, K_start, constraints) - x[xidx[i].L_fisch_research] -  x[xidx[i].L_nuss_research]
    constraints.update(i, CONSTRAINTIDX.Li_genuss, Li_genuss / L_max) # Randbedingung: Li_genuss(t) >= 0 <=> Li_genuss(t) / L_max >= 0
    return Li_genuss

# Berechung des Genuss-Anteils am Fischkonsum: Ci_fisch_genuss(t) = (1 - wi_fisch) * (Ci_fisch(t) - Ci_fisch_invest(t))
def calc_Ci_fisch_genuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Ci_fisch = calc_Ci_fisch(i, t, x, K_start, constraints)
    Ci_fisch_invest = calc_Ci_fisch_invest(i, t, x, K_start, constraints)
    Ci_fisch_genuss = (1 - x[xidx[i].w_fisch]) * (Ci_fisch - Ci_fisch_invest)
    return Ci_fisch_genuss

# Berechung des Genuss-Anteils am Nusskonsum: Ci_nuss_genuss(t) = (1 - wi_nuss) * (Ci_nuss(t) - Ci_nuss_invest(t))
def calc_Ci_nuss_genuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Ci_nuss = calc_Ci_nuss(i, t, x, K_start, constraints)
    Ci_nuss_invest = calc_Ci_nuss_invest(i, t, x, K_start, constraints)
    Ci_nuss_genuss = (1 - x[xidx[i].w_nuss]) * (Ci_nuss - Ci_nuss_invest)
    return Ci_nuss_genuss

# Berechung des gesamten Fischverbrauches: C1_fisch(t) = Y1_fisch(t) - tau1_fisch * Y1_fisch(t), C2_fisch(t) = Y2_fisch(t) + tau1_fisch * Y1_fisch(t)
def calc_Ci_fisch(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    assert i in (1, 2), f"## calc_Ci_fisch(i={i}): Ungültiger Parameter, i=1 (Robinson) oder i=2 (Freitag) erwartet"
    tau1_fisch = x[xidx[1].tau]
    Y1_fisch = calc_Yi_fisch(1, t, x, K_start, constraints)
    return (1 - tau1_fisch) * Y1_fisch if i == 1 else calc_Yi_fisch(2, t, x, K_start, constraints) + tau1_fisch * Y1_fisch

# Berechung des gesamten Nussverbrauches: C1_nuss(t) = Y1_nuss(t) + tau2_nuss * Y2_nuss(t), C2_nuss(t) = Y2_nuss(t) - tau2_nuss * Y2_nuss(t)
def calc_Ci_nuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    assert i in (1, 2), f"## calc_Ci_nuss(i={i}): Ungültiger Parameter, i=1 (Robinson) oder i=2 (Freitag) erwartet"
    tau2_nuss = x[xidx[2].tau]
    Y2_nuss = calc_Yi_nuss(2, t, x, K_start, constraints)
    return calc_Yi_nuss(1, t, x, K_start, constraints) + tau2_nuss * Y2_nuss if i == 1 else (1 - tau2_nuss) * Y2_nuss

# Berechung des Nutzens aus Subsistenzkonsum: Ui_subsistenz(t) = Ai_subsistenz * Li_subsistenz^alphai_subsistenz * Ci_fisch_subsistenz(t)^betai_subsistenz * Ci_nuss_subsistenz(t)^gammai_subsistenz
def calc_Ui_subsistenz(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Li_subsistenz = x[xidx[i].L_subsistenz]
    Ci_fisch_subsistenz = calc_Ci_fisch_subsistenz(i, t, x, K_start, constraints)
    Ci_nuss_subsistenz = calc_Ci_nuss_subsistenz(i, t, x, K_start, constraints)
    Ui_subsistenz = Ai_subsistenz[i] * Li_subsistenz**alphai_subsistenz[i] * Ci_fisch_subsistenz**betai_subsistenz[i] * Ci_nuss_subsistenz**gammai_subsistenz[i] if Ci_fisch_subsistenz >= 0 and Ci_nuss_subsistenz >= 0 else np.nan
    return Ui_subsistenz

# Berechung des Nutzens aus Genusskonsum:  Ui_genuss(t) = Ai_genuss * (Li_genuss(t) + Ci_fisch_genuss(t)/ri_fisch_genuss + Ci_nuss_genuss(t)/ri_nuss_genuss)^alphai_genuss
def calc_Ui_genuss(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Li_genuss = calc_Li_genuss(i, t, x, K_start, constraints)
    Ci_fisch_genuss = calc_Ci_fisch_genuss(i, t, x, K_start, constraints)
    Ci_nuss_genuss = calc_Ci_nuss_genuss(i, t, x, K_start, constraints)
    prod_faktor = Li_genuss + Ci_fisch_genuss/ri_fisch_genuss[i] + Ci_nuss_genuss/ri_nuss_genuss[i]
    Ui_genuss = Ai_genuss[i] * (prod_faktor**alphai_genuss[i]) if prod_faktor >= 0 else np.nan
    return Ui_genuss

# Berechung der Nutzenfunktion: Ui(i, t, x) = Ui_subsistenz(i, t, x) + Ui_genuss(i, t, x)
def calc_Ui(i: Literal[1, 2], t: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    assert i in (1, 2) and t >= 0 and len(x) == len(X1IDX) + len(X2IDX), f"## calc_Ui(i={i}, t={t}, len(x)={len(x)}): Ungültige Parameter, i=1 (Robinson) oder i=2 (Freitag), Zeit t>=0 und len(x)={len(X1IDX) + len(X2IDX)} erwartet"
    Ui_subsistenz = calc_Ui_subsistenz(i, t, x, K_start, constraints)
    Ui_genuss = calc_Ui_genuss(i, t, x, K_start, constraints)
    return Ui_subsistenz + Ui_genuss

# Berechung des Autarkienutzens: Ui_autarkie(i, t_autarkie, x_autarkie) = Ui(i, t_autarkie, x) mit x = x_autarkie plus tau1_fisch = 0 und tau2_nuss = 0
def calc_Ui_autarkie(i: Literal[1, 2], t_autarkie: float, x_autarkie: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    assert i in (1, 2) and t_autarkie >= 0 and len(x_autarkie) == len(X0IDX), f"## calc_Ui_autarkie(i={i}, t={t_autarkie}, len(x)={len(x_autarkie)}): Ungültige Parameter, i=1 (Robinson) oder i=2 (Freitag), Zeit t>=0 und len(x)={len(X0IDX)} erwartet"
    t = t_autarkie
    x = np.zeros(len(X1IDX) + len(X2IDX), dtype=np.float64)
    for n, x0idx in enumerate(X0IDX):
        x[xidx[i][x0idx.name]] = x_autarkie[x0idx.value] # x0idx.name = "L_fisch", "L_nuss", "L_subsistenz", usw., x0idx.value = 0, 1, 2, usw.
    Ui = calc_Ui(i, t, x, K_start, constraints)
    return Ui

# Berechung des diskontierten Nutzens: sum_{t_start<=t<t_max} (δ_i^t * Ui(t))
def calc_Ui_diskont(i: Literal[1, 2], delta_i: float, t_start: Union[int, float], x: np.ndarray, K_start: Capitals, constraints: Constraints) -> float:
    Ui_diskont = 0
    for t in range(t_start, t_max):
        Ui_diskont += (delta_i**t) * calc_Ui(i, t, x, K_start, constraints)
    return Ui_diskont

# Berechung des diskontierten Autarkienutzens: sum_{t_start<=t<t_max} (δ_i^t * Ui_autarkie(t))
def calc_Ui_autarkie_diskont(i: Literal[1, 2], delta_i: float, t_start: Union[int, float], x_autarkie: np.ndarray, K_start: Capitals):
    Ui_diskont = 0
    constraints = Constraints(np.inf)
    for t_autarkie in range(t_start, t_max):
        Ui_diskont += (delta_i**t_autarkie) * calc_Ui_autarkie(i, t_autarkie, x_autarkie, K_start, constraints)
    return Ui_diskont, constraints

# Maximierung des diskontierten Autarkienutzens: sum_{t_start<=t<t_max} (δ_i^t * Ui_autarkie(t)) -> max
# Rückgabewerte: sol_x, sol_U, sol_U_diskont, sol_constraints, sol_success, sol_usable
def maximize_Ui_autarkie_diskont(i: Literal[1, 2], delta_i: float, t_start: Union[int, float], x0: np.ndarray, K_start: Capitals, method = "COBYLA", min_constraints: float = -1e-8, verbose: int = 0):
    assert(len(x0) == len(X0IDX))
    Li_max_bound = L_max - Li_subsistenz_min
    opt_bounds = ((0, Li_max_bound), (0, Li_max_bound), (Li_subsistenz_min, L_max), (0, Li_max_bound), (0, Li_max_bound), (0, 1), (0, 1), (0, 1), (0, 1)) # i.e. 0<=Li_fisch<=Li_max_bound, 0<=Li_nuss<=Li_max_bound, L_subsistenz_min<=Li_subsistenz<=Li_max_bound, 0<=Li_fisch_research<=Li_max_bound,  0<=Li_nuss_research<=Li_max_bound, 0<=lambdai_fisch_invest<=1, 0<=lambdai_nuss_invest<=1, 0<=wi_fisch<=1, 0<=wi_nuss<=1
    # Cache um eine doppelte Berechnung von Optimierungs- und constraintsfunktion durch scipy.optimize.minimize zu vermeiden
    class MaxUiAutarkieOptimizerCache:
        def __init__(self):
            self.cached_x = None
            self.cached_result = None
            self.cached_constraints = None
            self.best_x = None
            self.best_result = None
            self.best_constraints = None
        def compute(self, x): # Führt calc_Ui_autarkie_diskont aus, falls vorher nicht mit x aufgerufen wurde
            if not np.array_equal(x, self.cached_x):
                self.cached_result, self.cached_constraints = calc_Ui_autarkie_diskont(i, delta_i, t_start, x, K_start)
                self.cached_result = -self.cached_result if np.isfinite(self.cached_result) else 1e100 # Ui_autarkie_diskont -> max <=> -Ui_autarkie_diskont -> min, Strafe falls nan
                self.cached_x = np.copy(x)
            if (self.best_result is None) or (self.cached_result < self.best_result and self.cached_constraints.min() >= min_constraints):
                self.best_x = self.cached_x
                self.best_result = self.cached_result
                self.best_constraints = self.cached_constraints
        def objective(self, x): # Rückgabe result_value (berechnet oder aus cache)
            self.compute(x)
            return self.cached_result
        def constraints(self, x):# Rückgabe constraints (berechnet oder aus cache)
            self.compute(x)
            return self.cached_constraints.autarkie_array(i)
    # Maximierung des diskontierten Autarkienutzens, Rückgabe x und Nutzen U unter Autarkie
    opt_cache = MaxUiAutarkieOptimizerCache()
    opt_cache.compute(x0)
    options = {"maxiter": 1000, "ftol": 1e-6} if method == "SLSQP" else {"maxiter": 5000, "rhobeg": 0.1, "tol": 1e-8, "catol": 1e-8} if method == "COBYLA" else None
    sol = scipy.optimize.minimize(fun = opt_cache.objective, x0 = x0, bounds = opt_bounds, method = method, options = options, constraints = [scipy.optimize.NonlinearConstraint(fun = opt_cache.constraints, lb = 0, ub = np.inf)])
    if sol.success: # Erfolg: Lösung des solvers zurückgeben
        sol_x = sol.x
        sol_U_diskont = -sol.fun
        sol_constraints = Constraints(np.inf)
        sol_U = calc_Ui_autarkie(i, t_start, sol_x, K_start, sol_constraints)
        if verbose > 0:
            print(f"  maximize_Ui_autarkie_diskont(i={i}, delta_i={delta_i:.2f}, t_start={t_start:.2f}, method={method}): x={fmt(sol_x,'{:.2f}')}, U={sol_U:.2f}, U_diskont={sol_U_diskont:.2f}, min(constraints)={sol_constraints.min():.2e}, success=True")
        if verbose >= 0 and sol_constraints.min() < min_constraints:
            print(f"  maximize_Ui_autarkie_diskont(i={i}, delta_i={delta_i:.2f}, t_start={t_start:.2f}, method={method}) fehlgeschlagen: Lösung verletzt Randbedingungen, constraints={sol_constraints.autarkie_array(i)} >= 0 erwartet")
        return sol_x, sol_U, sol_U_diskont, sol_constraints, True, True
    # Kein Erfolg, sol.success == False: sol.x ist der letzte Kandidat vor Abbruch; der Kandidat kann trotzdem ein brauchbares Ergebnis sein, solange der Kandidat die Randbedingungen erfüllt
    # Wir verwenden die beste bei der Suche gefundene Lösung; die beste Lösung ist verwendbar, sofern die Randbedingungen erfüllt sind (constraints.min >= min_constraints)
    if verbose >= 0:
        nit = sol.nit if hasattr(sol,"nit") else None
        print(f"  maximize_Ui_autarkie_diskont(i={i}, delta_i={delta_i:.2f}, t_start={t_start:.2f}, method={method}): scipy.optimize.minimize failed, message:\"{sol.message}\", status={sol.status}, nit={nit}, nfev={sol.nfev}, fun={sol.fun}, max(abs(x-x0)={np.max(np.abs(sol.x - x0))}, x0={fmt(x0,'{:.2f}')}, best_x={fmt(opt_cache.best_x,'{:.2f}')}, best_constraints={opt_cache.best_constraints.autarkie_array(i)}, min(best_constraints)={opt_cache.best_constraints.min():.2e}")
    sol_x = opt_cache.best_x
    sol_U_diskont = -opt_cache.best_result
    sol_usable = opt_cache.best_constraints.min() >= min_constraints
    sol_constraints = Constraints(np.inf)
    sol_U = calc_Ui_autarkie(i, t_start, sol_x, K_start, sol_constraints)
    if verbose > 0:
        print(f"  maximize_Ui_autarkie_diskont(i={i}, delta_i={delta_i:.2f}, t_start={t_start:.2f}, method={method}): x={fmt(sol_x,'{:.2f}')}, U={sol_U:.2f}, U_diskont={sol_U_diskont:.2f}, min(constraints)={sol_constraints.min():.2e}, success=False, usable={sol_usable}")
    return sol_x, sol_U, sol_U_diskont, sol_constraints, False, sol_usable

# Maximierung des diskontierten Autarkienutzens: sum_{t_start<=t<t_max} (δ_i^t * Ui_autarkie(t)) -> max, mit "COBYLA" und "SLSQP" und Rückgabe der besten Lösung
def maximize_Ui_autarkie_diskont_best_method(i: Literal[1, 2], delta_i: float, t_start: Union[int, float], x0: np.ndarray, K_start: Capitals, min_constraints: float = -1e-8, verbose: int = 0):
    xi_autarkie_COBYLA, Ui_autarkie_COBYLA, Vi_autarkie_COBYLA, constraints_autarkie_COBYLA, sol_success_COBYLA, sol_usable_COBYLA = maximize_Ui_autarkie_diskont(i, delta_i, t_start, x0, K_start, method="COBYLA", min_constraints=min_constraints, verbose=-1)
    xi_autarkie_SLSQP, Ui_autarkie_SLSQP, Vi_autarkie_SLSQP, constraints_autarkie_SLSQP, sol_success_SLSQP, sol_usable_SLSQP = maximize_Ui_autarkie_diskont(i, delta_i, t_start, x0, K_start, method="SLSQP", min_constraints=min_constraints, verbose=-1)
    if sol_success_COBYLA and sol_success_SLSQP:
        use_SLSQP = (Vi_autarkie_SLSQP >= Vi_autarkie_COBYLA)
    elif sol_success_SLSQP:
        use_SLSQP = True
    elif sol_success_COBYLA:
        use_SLSQP = False
    elif sol_usable_SLSQP:
        use_SLSQP = True
    elif sol_usable_COBYLA:
        use_SLSQP = False
    else:
        use_SLSQP = True # Fallback wenn beide scheitern
    if use_SLSQP:
        xi_autarkie, Ui_autarkie, Vi_autarkie, constraints_autarkie, sol_success, sol_usable, method_str = xi_autarkie_SLSQP, Ui_autarkie_SLSQP, Vi_autarkie_SLSQP, constraints_autarkie_SLSQP, sol_success_SLSQP, sol_usable_SLSQP, "SLSQP"
    else:
        xi_autarkie, Ui_autarkie, Vi_autarkie, constraints_autarkie, sol_success, sol_usable, method_str = xi_autarkie_COBYLA, Ui_autarkie_COBYLA, Vi_autarkie_COBYLA, constraints_autarkie_COBYLA, sol_success_COBYLA, sol_usable_COBYLA, "COBYLA"
    if verbose > 0: 
        print(f"  Autarkienutzen {i} (δ{i}={delta_i:.2f}, t_start={t_start}, method={method_str}): x{i}_autarkie={fmt(xi_autarkie,'{:.3f}')}, U{i}_autarkie={Ui_autarkie:.2f}, V{i}_autarkie={Vi_autarkie:.4e}, min(constraints{i})={constraints_autarkie.min():.2e}, success={sol_success}, usable={sol_usable}")
    if not sol_success and sol_usable and constraints_autarkie.min() >= min_constraints:
        print(f"  maximize_Ui_autarkie_diskont_best_method(δ{i}={delta_i:.2f}, t_start={t_start}, K0={K_start}): Maximierung der diskontierten Autarkienutzen suboptimal, success={sol_success}, usable={sol_usable}, min(constraints)={constraints_autarkie.min():.2e}")    
    elif not sol_success or constraints_autarkie.min() < min_constraints:
        print(f"  ## maximize_Ui_autarkie_diskont_best_method(δ{i}={delta_i:.2f}, t_start={t_start}, K0={K_start}): Maximierung der diskontierten Autarkienutzen fehlgeschlagen, success={sol_success}, usable={sol_usable}, min(constraints)={constraints_autarkie.min():.2e}")
    return xi_autarkie, Ui_autarkie, Vi_autarkie, constraints_autarkie, sol_success, sol_usable

# Berechnung des logarithmischen Nashproduktes NP = log(V1 - V1_autarkie) + log(V2 - V2_autarkie) mit den diskontierten Nutzen Vi und den diskontierten Autarkienutzen Vi_autarkie
def calc_nash_product_log(delta_1: float, delta_2: float, t_start: Union[int, float], x: np.ndarray, K_start: Capitals, V1_autarkie: float, V2_autarkie: float, verbose: int = 0):
    constraints = Constraints(np.inf)
    V1 = calc_Ui_diskont(1, delta_i=delta_1, t_start=t_start, x=x, K_start=K_start, constraints=constraints)
    V2 = calc_Ui_diskont(2, delta_i=delta_2, t_start=t_start, x=x, K_start=K_start, constraints=constraints)
    # Randbedingungen: Vi - Vi_autarkie >= epsilon <=> (Vi - Vi_autarkie - epsilon) / max(1, |Vi_autarkie|) >= 0
    Vi_max = max(1.0, abs(V1_autarkie), abs(V2_autarkie))
    eps_surplus = 1e-8 * Vi_max
    constraints.update(1, CONSTRAINTIDX.Vi, (V1 - V1_autarkie - eps_surplus) / Vi_max)
    constraints.update(2, CONSTRAINTIDX.Vi, (V2 - V2_autarkie - eps_surplus) / Vi_max)
    if V1 - V1_autarkie < eps_surplus or V2 - V2_autarkie < eps_surplus:
        nash_product = np.log(max(V1 - V1_autarkie, eps_surplus * 1e-3)) + np.log(max(V2 - V2_autarkie, eps_surplus * 1e-3)) # Statt np.nan: Strafe bei Verletzung der Randbedingung V1 > V1_autarkie, V2 > V2_autarkie
    else:
        nash_product = np.log(V1 - V1_autarkie) + np.log(V2 - V2_autarkie) # (V1 - V1_autarkie) * (V2 - V2_autarkie)
    if verbose > 0:
        print(f"  calc_nash_product_log(δ1={delta_1:.2f}, δ2={delta_2:.2f}, t_start={t_start:.2f}, K0={K_start}, x={fmt(x,'{:.2f}')}): δV1={V1-V1_autarkie:.2f}, δV2={V2-V2_autarkie:.2f}, NP={nash_product:.8f}, min(constraints)={constraints.min():.2e}")
    return nash_product, constraints

# Maximierung Nashprodukt log(V1 - V1_autarkie) + log(V2 - V2_autarkie) → max mit gegebenem Startwert x0 und Autarkienutzen V1_autarkie, V2_autarkie
def maximize_nash_product_x0(delta_1: float, delta_2: float, t_start: Union[int, float], x0: np.ndarray, K_start: Capitals, V1_autarkie: float, V2_autarkie: float, min_constraints: float = -1e-8, check_x0_constraints: bool = True, verbose: int = 0):
    if verbose > 0: 
        print(f"  Maximierung Nashprodukt (δ1={delta_1:.2f}, δ2={delta_2:.2f}, t={t_start}, K0={K_start}, x0={fmt(x0,'{:.2f}')}):")
    if check_x0_constraints: # Randbedingungen mit Startwert x0 zulässig?
        nash_product, constraints = calc_nash_product_log(delta_1, delta_2, t_start, x0, K_start, V1_autarkie, V2_autarkie)
        if verbose > 1:
            print(f"  x=x0: log(NP)={nash_product:.2f}, constraints={fmt(constraints.array(),'{:.2e}')}, min(constraints)={constraints.min():.2e}")
        if constraints.min() < min_constraints:
            print(f"  Startwerte für Maximierung Nashprodukt ungünstig, x0={fmt(x0,'{:.3f}')} verletzt Randbedingungen, min(constraints)={constraints.min():.2e}")
    # Cache um eine doppelte Berechnung von Optimierungs- und constraintsfunktion durch scipy.optimize.minimize zu vermeiden
    class MaxNashproductOptimizerCache:
        def __init__(self):
            self.cached_x = None
            self.cached_result = None
            self.cached_constraints = None
            self.best_x = None
            self.best_result = None
            self.best_constraints = None
        def compute(self, x): # Führt calc_nash_product_log aus, falls vorher nicht mit x aufgerufen wurde
            if not np.array_equal(x, self.cached_x):
                self.cached_result, self.cached_constraints = calc_nash_product_log(delta_1, delta_2, t_start, x, K_start, V1_autarkie, V2_autarkie)
                self.cached_result = -self.cached_result if np.isfinite(self.cached_result) else 1e100 # NP -> max <=> -NP -> min, Strafe falls nan
                self.cached_x = np.copy(x)
            if (self.best_result is None) or (self.cached_result < self.best_result and self.cached_constraints.min() >= min_constraints):
                self.best_x = self.cached_x
                self.best_result = self.cached_result
                self.best_constraints = self.cached_constraints
        def objective(self, x): # Rückgabe log. Nashprodukt (berechnet oder aus cache)
            self.compute(x)
            return self.cached_result
        def constraints(self, x): # Rückgabe constraints (berechnet oder aus cache, Verletzung falls constraints < 0)
            self.compute(x)
            return self.cached_constraints.array()
    # bounds: 0<=Li_fisch<=Li_max_bound, 0<=Li_nuss<=Li_max_bound, L_subsistenz_min<=Li_subsistenz<=Li_max_bound, 0<=Li_fisch_research<=Li_max_bound,  0<=Li_nuss_research<=Li_max_bound, 0<=lambdai_fisch_invest<=1, 0<=lambdai_nuss_invest<=1, 0<=wi_fisch<=1, 0<=wi_nuss<=1, 0<=taui<=1
    Li_max_bound = L_max - Li_subsistenz_min
    opt_bounds = ((0, Li_max_bound), (0, Li_max_bound), (Li_subsistenz_min, L_max), (0, Li_max_bound), (0, Li_max_bound), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, Li_max_bound), (0, Li_max_bound), (Li_subsistenz_min, L_max), (0, Li_max_bound), (0, Li_max_bound), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)) 
    opt_cache = MaxNashproductOptimizerCache()
    opt_cache.compute(x0)
    # Default: scipy.optimize.minimize mit SLSQP
    opt_method = "SLSQP" # SLSQP options are: ftol (float): Precision target for the objective function value, default: 1e-6, maxiter (int): Maximum number of iterations, default: 100, eps (float): Step size for the Jacobian matrix
    options = {"SLSQP": {"maxiter": 1000, "ftol": 1e-6}, "COBYLA": {"maxiter": 5000, "rhobeg": 0.1, "tol": 1e-8, "catol": 1e-8}}
    sol = scipy.optimize.minimize(fun = opt_cache.objective, x0 = x0, bounds = opt_bounds, method = opt_method, options = options[opt_method], constraints = [scipy.optimize.NonlinearConstraint(fun = opt_cache.constraints, lb = 0, ub = np.inf)])
    sol_x = sol.x if sol.success else opt_cache.best_x
    sol_V = -sol.fun if sol.success else opt_cache.best_result
    sol_usable = True if sol.success else opt_cache.best_constraints.min() >= min_constraints
    # Falls Optimierung mit SLSQP fehlschlägt: zusätzlich mit COBYLA und bessere Lösung verwenden
    if not sol.success:
        opt_cache.cached_x = None
        sol_COBYLA = scipy.optimize.minimize(fun = opt_cache.objective, x0 = x0, bounds = opt_bounds, method = "COBYLA", options = options["COBYLA"], constraints = [scipy.optimize.NonlinearConstraint(fun = opt_cache.constraints, lb = 0, ub = np.inf)])
        if sol_COBYLA.success or (opt_cache.best_result < -sol_V and opt_cache.best_constraints.min() >= min_constraints):
            sol = sol_COBYLA
            sol_x = sol.x if sol.success else opt_cache.best_x
            sol_V = -sol.fun if sol.success else opt_cache.best_result
            sol_usable = True if sol.success else opt_cache.best_constraints.min() >= min_constraints
            opt_method = "COBYLA"
            print(f"  maximize_nash_product_x0: scipy.optimize.minimize mit SLSQP fehlgeschlagen, mit COBYLA erfolgreich")
        else:
            print(f"  maximize_nash_product_x0: scipy.optimize.minimize mit SLSQP und COBYLA fehlgeschlagen (SLSQP: {sol.message}, COBYLA: {sol_COBYLA.message}")
    # Beste Lösung zurückgeben
    sol_logNP, sol_constraints_NP = calc_nash_product_log(delta_1, delta_2, t_start, sol_x, K_start, V1_autarkie, V2_autarkie)
    sol_constraints_U1 = Constraints(np.inf)
    sol_constraints_U2 = Constraints(np.inf)
    sol_U1 = calc_Ui(1, 0, sol_x, K_start, sol_constraints_U1)
    sol_U2 = calc_Ui(2, 0, sol_x, K_start, sol_constraints_U2)
    sol_constraints_min = min(sol_constraints_NP.min(), sol_constraints_U1.min(), sol_constraints_U2.min())
    if sol.success or sol_usable:
        if verbose > 0:
            diagnostic_constraints = Constraints(np.inf)
            lambdas = [sol_x[xidx[1].lambda_fisch_invest], sol_x[xidx[1].lambda_nuss_invest], sol_x[xidx[2].lambda_fisch_invest], sol_x[xidx[2].lambda_nuss_invest]]
            L_invests = [calc_Li_fisch_invest(1, t_start, sol_x, K_start, diagnostic_constraints), calc_Li_nuss_invest(1, t_start, sol_x, K_start, diagnostic_constraints), calc_Li_fisch_invest(2, t_start, sol_x, K_start, diagnostic_constraints), calc_Li_nuss_invest(2, t_start, sol_x, K_start, diagnostic_constraints)]
            L_researchs = [sol_x[xidx[1].L_fisch_research], sol_x[xidx[1].L_nuss_research], sol_x[xidx[2].L_fisch_research], sol_x[xidx[2].L_nuss_research]]
            K1_t_max = [calc_Ki_fisch(1, t_max-1, sol_x, K_start, diagnostic_constraints), calc_Ki_nuss (1, t_max-1, sol_x, K_start, diagnostic_constraints), calc_Ri_fisch(1, t_max-1, sol_x, K_start, diagnostic_constraints), calc_Ri_nuss (1, t_max-1, sol_x, K_start, diagnostic_constraints)]
            K2_t_max = [calc_Ki_fisch(2, t_max-1, sol_x, K_start, diagnostic_constraints), calc_Ki_nuss (2, t_max-1, sol_x, K_start, diagnostic_constraints), calc_Ri_fisch(2, t_max-1, sol_x, K_start, diagnostic_constraints), calc_Ri_nuss (2, t_max-1, sol_x, K_start, diagnostic_constraints)]
            print(f"  x={fmt(sol_x,'{:.2f}')}, lambda={fmt(lambdas,'{:.2e}')}, L_invest={fmt(L_invests,'{:.2f}')}, L_research={fmt(L_researchs,'{:.2f}')}, success={sol.success}, usable={sol_usable}")
            print(f"  K1(t={t_start+t_max-1})={fmt(K1_t_max,'{:.2f}')}, K2(t={t_start+t_max-1})={fmt(K2_t_max,'{:.2f}')}, Σ(L_invest(t={t_start}))={sum(L_invests):.3f}, Σ(L_research)={sum(L_researchs):.3f}, U1={sol_U1:.2f}, U2={sol_U2:.2f}, log(NP)={sol_logNP:.2f}, NP={np.exp(sol_logNP):.2e}, min(constraints)={sol_constraints_min:.2e}")
            if diagnostic_constraints.min() < min_constraints:
                print(f"  Diagnostic verletzt Randbedingungen, min(constraints)={diagnostic_constraints.min():.2e}")
        if sol_constraints_min < min_constraints:
            print(f"  Lösung verletzt Randbedingungen, min(constraints)={sol_constraints_min:.2e}, constraints_NP={sol_constraints_NP.array()}, constraints_U1={sol_constraints_U1.array()}, constraints_U2={sol_constraints_U2.array()}")
    else:
        nit = f"nit={sol.nit}, " if hasattr(sol,"nit") else ""
        maxcv = f"maxcv={sol.maxcv}, " if hasattr(sol,"maxcv") else ""
        print(f"  ## scipy.optimize.minimize mit method={opt_method}) fehlgeschlagen: \"{sol.message}\", status={sol.status}, {nit}{maxcv}nfev={sol.nfev}, fun={sol.fun}, max(abs(x-x0))={np.max(np.abs(sol.x - x0))}")
    return sol_x, sol_U1, sol_U2, sol_logNP, sol_constraints_min, sol.success, sol_usable

# Berechnung des Nashlösung: 
# Vi = sum_{0<=t<t_max} (δ_i^t * Ui(t)) # Diskontierter Nutzen
# Vi_autarkie = sum_{0<=t<t_max} (δ_i^t * Ui_autarkie(t)) # Diskontierter Autarkienutzen
# (V1 - V1_autarkie) * (V2 - V2_autarkie) → max mit den Randbedingungen V1 >= V1_autarkie und V2 >= V2_autarkie
# log(V1 - V1_autarkie) + log(V2 - V2_autarkie) → max mit den Randbedingungen V1 - V1_autarkie >= ε und V2 - V2_autarkie >= ε
def maximize_nash_product(settings: Settings, t_start: Union[int, float], K_start: Capitals, min_constraints: float = -1e-8, verbose: int = 0):

    # Maximierung der diskontierten Autarkienutzen
    x1_autarkie, U1_autarkie, V1_autarkie, constraints1_autarkie, sol1_success, sol1_usable = maximize_Ui_autarkie_diskont_best_method(i = 1, delta_i = settings.delta_1, t_start = t_start, x0 = settings.x0_autarkie_1, K_start = K_start, min_constraints = min_constraints, verbose = verbose)
    x2_autarkie, U2_autarkie, V2_autarkie, constraints2_autarkie, sol2_success, sol2_usable = maximize_Ui_autarkie_diskont_best_method(i = 2, delta_i = settings.delta_2, t_start = t_start, x0 = settings.x0_autarkie_2, K_start = K_start, min_constraints = min_constraints, verbose = verbose)
    # Test der Stabilität der Maximierung der diskontierten Autarkienutzen gegen Startwerte:
    # x1_autarkie, U1_autarkie, V1_autarkie, constraints1_autarkie = search_Ui_autarkie_diskont(1, delta_1, 0)
    # x2_autarkie, U2_autarkie, V2_autarkie, constraints2_autarkie = search_Ui_autarkie_diskont(2, delta_2, 0)
    
    # Maximierung Nashprodukt log(V1 - V1_autarkie) + log(V2 - V2_autarkie) → max mit Randbedingungen V1 - V1_autarkie >= ε und V2 - V2_autarkie >= ε und Startwert x0:
    # x0 = [ L1_fisch, L1_nuss, L1_subsistenz, L1_fisch_research, L1_nuss_research, lambda1_fisch_invest, lambda1_nuss_invest, w1_fisch, w1_nuss, tau1_fisch, 
    #        L2_fisch, L2_nuss, L2_subsistenz, L2_fisch_research, L2_nuss_research, lambda2_fisch_invest, lambda2_nuss_invest, w2_fisch, w2_nuss, tau2_nuss ]
    # x0 = [ x1_autarkie, tau1_fisch=0.1, x2_autarkie, tau2_nuss=0.1 ] 
    x0 = np.concatenate((x1_autarkie, [0.1], x2_autarkie, [0.1])) if settings.x0 is None else settings.x0
    x, U1, U2, logNP, constraint, success, usable = maximize_nash_product_x0(settings.delta_1, settings.delta_2, t_start, x0, K_start, V1_autarkie, V2_autarkie, min_constraints, False, verbose)
    # Test Stabilität gegen Startwerte:
    # test_maximize_nash_product(delta_1, delta_2, x1_autarkie, x2_autarkie, V1_autarkie, V2_autarkie)
    
    return MaxNPResult(x, x1_autarkie, V1_autarkie, x2_autarkie, V2_autarkie, U1, U2, logNP, constraint, success and sol1_success and sol2_success, usable and sol1_usable and sol2_usable)

# Kalkulation der Kennwerte (Produktion Yi_fisch, Yi_nuss, Kapitalwerte Ki_fisch, Ki_nuss, Ri_fisch, Ri_nuss und Nutzen U1, U2) über der Zeit t
def calc_key_figures(x: np.ndarray, period_start: Union[int, float], period_end: Union[int, float], K_akkumulated: Capitals, key_figures: KeyFigures):
    period_duration = period_end - period_start
    constraints = Constraints(np.inf)
    for t in range(period_duration):
        key_figures.t_values = np.append(key_figures.t_values, t + period_start)
        # Kennwerte
        for i in [1, 2]:
            key_figures.Yi_fisch_values[i] = np.append(key_figures.Yi_fisch_values[i], calc_Yi_fisch(i, t, x, K_akkumulated, constraints))
            key_figures.Yi_nuss_values[i]  = np.append(key_figures.Yi_nuss_values[i],  calc_Yi_nuss (i, t, x, K_akkumulated, constraints))
            key_figures.Ki_fisch_values[i] = np.append(key_figures.Ki_fisch_values[i], calc_Ki_fisch(i, t, x, K_akkumulated, constraints))
            key_figures.Ki_nuss_values[i]  = np.append(key_figures.Ki_nuss_values[i],  calc_Ki_nuss (i, t, x, K_akkumulated, constraints))
            key_figures.Ri_fisch_values[i] = np.append(key_figures.Ri_fisch_values[i], calc_Ri_fisch(i, t, x, K_akkumulated, constraints))
            key_figures.Ri_nuss_values[i]  = np.append(key_figures.Ri_nuss_values[i],  calc_Ri_nuss (i, t, x, K_akkumulated, constraints))
            key_figures.Ui_values[i]       = np.append(key_figures.Ui_values[i],       calc_Ui      (i, t, x, K_akkumulated, constraints))
        # Tauschpreis in Nüssen pro Fisch: tau_nuts_per_fish = (tau2_nuss * Y2_nuss(t)) / (tau1_fisch * Y1_fisch(t))
        Y1_fisch = calc_Yi_fisch(1, t, x, K_akkumulated, constraints)
        Y2_nuss = calc_Yi_nuss (2, t, x, K_akkumulated, constraints)
        tau1_fisch = x[X1IDX.tau]
        tau2_nuss = x[X2IDX.tau]
        tau_nuts_per_fish = (tau2_nuss * Y2_nuss) / (tau1_fisch * Y1_fisch) if tau1_fisch * Y1_fisch > 1e-6 else 0
        key_figures.tau_nuts_per_fish = np.append(key_figures.tau_nuts_per_fish, tau_nuts_per_fish)
    # Akkumulation der Kapitalwerte Ki_fisch, Ki_nuss, Ri_fisch, Ri_nuss (Startwerte für die nächste Periode)
    K_orig = K_akkumulated.copy()
    for i in [1, 2]:
        K_akkumulated.Ki_fisch[i] = calc_Ki_fisch(i, period_duration, x, K_orig, constraints)
        K_akkumulated.Ki_nuss[i]  = calc_Ki_nuss (i, period_duration, x, K_orig, constraints)
        K_akkumulated.Ri_fisch[i] = calc_Ri_fisch(i, period_duration, x, K_orig, constraints)
        K_akkumulated.Ri_nuss[i]  = calc_Ri_nuss (i, period_duration, x, K_orig, constraints)
    if constraints.min() < -1e-6:
        print(f"  ## mini_wm_robinson_two_persons(δ1={setting.delta_1:.2f}, δ2={setting.delta_2:.2f}, K0={K_akkumulated}, period_start={period_start}, period_end={period_end}): Verletzung Randbedingungen, min(constraints)={constraints.min():.3e}")

# Plot Produktion Y_fisch(t) = Y1_fisch(t)+Y2_fisch(t), Y_nuss(t) = Y1_nuss(t)+Y1_nuss(t) und Kapital K(t) = K1_fisch(t)+K2_fisch(t)+K1_nuss(t)+K2_nuss(t) über Zeit t
def plot_key_figures(key_figures: KeyFigures, title: str, plot_log: bool):
    from mini_wm_robinson_race_to_preempt import rcw_02_plot_dbg, save_plot_png
    global plt_cnt
    t_values = key_figures.t_values
    Y_fisch_values = key_figures.Yi_fisch_values[1] + key_figures.Yi_fisch_values[2]
    Y_nuss_values = key_figures.Yi_nuss_values[1] + key_figures.Yi_nuss_values[2]
    K_values = key_figures.Ki_fisch_values[1] + key_figures.Ki_fisch_values[2] + key_figures.Ki_nuss_values[1] + key_figures.Ki_nuss_values[2]
    R_values = key_figures.Ri_fisch_values[1] + key_figures.Ri_fisch_values[2] + key_figures.Ri_nuss_values[1] + key_figures.Ri_nuss_values[2]
    P_values = np.copy(key_figures.tau_nuts_per_fish)
    # Preise skalieren zwecks Darstellung
    max_P_val = P_values.max()
    max_y_val = max(max_P_val, Y_fisch_values.max(), Y_nuss_values.max(), K_values.max(), R_values.max())
    P_scale = int((max_y_val / 10) / max_P_val) if max_P_val > 1e-6 and max_P_val < max_y_val / 10 and plot_log == False else 1
    rcw_02_plot_dbg(plots=[(t_values, Y_fisch_values, "Y_fisch"), (t_values, Y_nuss_values, "Y_nuss"), (t_values, K_values, "K"), (t_values, R_values, "R"), (t_values, P_values*P_scale, f"Preis*{P_scale}")], xlabel="t", ylabel="Produktion Y, phys.Kapital K, technol.Kapital R", block=False)
    if plot_log:
        plt.yscale("log")
    plt.title(title)
    plt_cnt += 1
    save_plot_png(f"{plt_now_str}_mini_wm_rcw_02_2persons_{plt_cnt:02d}.png")
    with open(f"{plt_now_str}_mini_wm_rcw_02_2persons_{plt_cnt:02d}.json", "w") as json_file:
        json.dump(key_figures.to_json(), json_file)

if __name__ == "__main__":
    print(f"mini_wm_robinson_two_persons:")
    start_time = time.perf_counter()
    max_np_counter = 0
    settings = [Settings(delta_1 = 0.1, delta_2 = 0.1, x0_autarkie_1 = np.array([1, 1,  8, 0, 0, 0, 0, 1, 1], dtype=np.float64), x0_autarkie_2 = np.array([1, 1,  8, 0, 0, 0, 0, 1, 1], dtype=np.float64), x0 = None, plot_log = True),
                Settings(delta_1 = 0.9, delta_2 = 0.9, x0_autarkie_1 = np.array([1, 1, 20, 0, 0, 0, 0, 1, 1], dtype=np.float64), x0_autarkie_2 = np.array([1, 1, 20, 0, 0, 0, 0, 1, 1], dtype=np.float64), x0 = None, plot_log = True),
                Settings(delta_1 = 1.0, delta_2 = 1.0, x0_autarkie_1 = np.array([1, 1, 22, 0, 0, 0, 0, 1, 1], dtype=np.float64), x0_autarkie_2 = np.array([1, 1, 22, 0, 0, 0, 0, 1, 1], dtype=np.float64), x0 = None, plot_log = True)]
    if True: # Robinson und Freitag verhandeln einmal am Anfang: x konstant über der Zeit t, d.h. Allokation, Tauschmengen, Investitionen konstant, Kapital akkumuliert
        for setting in settings:
            # Maximierung Nashprodukt
            print(f"")
            K = Capitals()
            key_figures = KeyFigures(delta_1 = setting.delta_1, delta_2 = setting.delta_2)
            max_np_result = maximize_nash_product(settings = setting, t_start = 0, K_start = K, min_constraints = -1e-6, verbose = 1)
            max_np_counter += 1
            # Kalkulation und plot der Kennwerte (Produktion und Kapital) über der Zeit t
            calc_key_figures(max_np_result.x, 0, t_max, K, key_figures)
            plot_key_figures(key_figures, f"Produktion Y_fisch, Y_nuss, phys.Kapital K und technol.Kapital R über Zeit t für δ1={key_figures.deltai[1]:.2f}, δ2={key_figures.deltai[2]:.2f}", setting.plot_log)

    if False: # Neuverhandlung nach jedem (n-ten) Zeitschritt, mit diskontiertem Nutzen jeweils über t_max Schritte, sehr rechenaufwendig
        # max_np_time_interval = t_max: Robinson und Freitag verhandeln nur einmal am Anfang
        # max_np_time_interval = 1: Robinson und Freitag verhandeln nach jedem Zeitschritt neu (sehr rechenaufwendig)
        # max_np_time_interval = 10: Robinson und Freitag verhandeln nach jedem 10. Zeitschritt neu (rechenaufwendig, Abschätzung der ungefähren Entwicklung von Allokation und Preisen bei wachsendem Kapital)
        max_np_time_interval = 1 # Maximierung des Nashproduktes in jedem Zeitschritt
        for setting in settings:
            x0 = setting.x0
            x0_autarkie_1 = setting.x0_autarkie_1
            x0_autarkie_2 = setting.x0_autarkie_2
            K_akkumulated = Capitals()
            key_figures = KeyFigures(delta_1 = setting.delta_1, delta_2 = setting.delta_2)
            for period_start in range(0, t_max, max_np_time_interval):
                period_end = min(period_start + max_np_time_interval, t_max)
                setting.x0 = x0
                setting.x0_autarkie_1 = x0_autarkie_1
                setting.x0_autarkie_2 = x0_autarkie_2
                # Maximierung Nashprodukt
                print(f"\n  δ1={setting.delta_1:.2f}, δ2={setting.delta_2:.2f}, Periode {period_start}-{period_end}:")
                max_np_result = maximize_nash_product(settings = setting, t_start = 0, K_start = K_akkumulated, min_constraints = -1e-5, verbose = 1)
                max_np_counter += 1
                if max_np_result.success or max_np_result.usable: # Maximierung Nashprodukt erfolgreich
                    # Kalkulation der Kennwerte (Produktion Yi_fisch, Yi_nuss, Kapitalwerte Ki_fisch, Ki_nuss, Ri_fisch, Ri_nuss und Nutzen U1, U2) über der Zeit t
                    calc_key_figures(max_np_result.x, period_start, period_end, K_akkumulated, key_figures)
                    # Lösung als Startwert für nächste Periode setzen
                    x0 = np.copy(max_np_result.x)
                    x0_autarkie_1 = np.copy(max_np_result.x1_autarkie)
                    x0_autarkie_2 = np.copy(max_np_result.x2_autarkie)
                else: # Diagnose nach Misserfolg
                    print(f"  ## mini_wm_robinson_two_persons(δ1={setting.delta_1:.2f}, δ2={setting.delta_2:.2f}, K0={K_akkumulated}, period_start={period_start}, period_end={period_end}): maximize_nash_product failed")
                    print(f"  ## Diagnosis:")
                    for method in ["COBYLA", "SLSQP"]:
                        x1_autarkie, U1_autarkie, V1_autarkie, constraints1_autarkie, sol_success1_autarkie, sol_usable1_autarkie = maximize_Ui_autarkie_diskont(1, setting.delta_1, 0, x0_autarkie_1, K_akkumulated, method, -1e-8, verbose=1)
                        x2_autarkie, U2_autarkie, V2_autarkie, constraints2_autarkie, sol_success2_autarkie, sol_usable2_autarkie = maximize_Ui_autarkie_diskont(2, setting.delta_2, 0, x0_autarkie_2, K_akkumulated, method, -1e-8, verbose=1)
                        maximize_nash_product_x0(setting.delta_1, setting.delta_2, 0, x0, K_akkumulated, V1_autarkie, V2_autarkie, -1e-8, True, verbose=1)
                    break
            # Plot Produktion Y_fisch(t) = Y1_fisch(t)+Y2_fisch(t), Y_nuss(t) = Y1_nuss(t)+Y1_nuss(t) und Kapital K(t) = K1_fisch(t)+K2_fisch(t)+K1_nuss(t)+K2_nuss(t) über Zeit t
            plot_key_figures(key_figures, f"Produktion Y_fisch, Y_nuss, phys.Kapital K und technol.Kapital R über Zeit t für δ1={key_figures.deltai[1]:.2f}, δ2={key_figures.deltai[2]:.2f}, Intervall={max_np_time_interval}", setting.plot_log)

    execution_time_sec = time.perf_counter() - start_time
    print(f"\nmini_wm_robinson_two_persons finished, calculated {max_np_counter} nash solutions in {execution_time_sec:.3f} sec, {(execution_time_sec/max_np_counter):.3f} sec per solution")
    plt.show()
