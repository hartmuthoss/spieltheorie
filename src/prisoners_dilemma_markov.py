"""
Das unendlich oft wiederholte Gefangenendilemma mit Markov-Prozess:

Beide Gefangene spielen Tit for Tat mit Antesten: Jeder Spieler $S_i$ ist im Ausgangspunkt kooperativ, testet den Gegenspieler aber mit einer Wahrscheinlichkeit $q_i$ durch Verrat an. 
Reagiert der Gegenspieler mit Bestrafung, wird einmalig kooperiert, ansonsten mit Tit for Tat reagiert. Beide starten mit Kooperation. 

Der Zustandsraum ist { (L,L), (L,G), (G,L), (G,G) }, L=Lügen (Kooperation), G=Gestehen (Verrat, Strafe).
Die Zustandsübergänge in prisoners_dilemma_repeated_05b_markov.svg:

Sofortnutzen:  
| **Zustand**                      | **Nutzen ($U_1$, $U_2$)** |
| -------------------------------- | ------------------------- |
| Z1, (L, L): (Leugnen, Leugnen)   |  $u_1 = (R, R) = (3, 3)$  |
| Z2, (L, G): (Leugnen, Gestehen)  |  $u_2 = (S, T) = (1, 4)$  |
| Z3, (G, L): (Gestehen, Leugnen)  |  $u_3 = (T, S) = (4, 1)$  |
| Z4, (G, G): (Gestehen, Gestehen) |  $u_4 = (P, P) = (2, 2)$  |
| Z5, (L, G): (Leugnen, Gestehen)  |  $u_5 = (S, T) = (1, 4)$  |
| Z6, (G, L): (Gestehen, Leugnen)  |  $u_6 = (T, S) = (4, 1)$  |

Sofortnutzen Spieler 1: $r_1 = [ \ R \ S \ T \ P \ S \ T \ ]^T$  
Sofortnutzen Spieler 2: $r_2 = [ \ R \ T \ S \ P \ T \ S \ ]^T$  

Diskontierte Nutzenfunktionen:  
$U = U(Z1,t=0)$  
$U(Z1,t) = δ^t (R, R) + (1-q_1) (1-q_2) U(Z1,t+1) + q_2 (1-q_1) U(Z2,t+1) + q_1 (1-q_2) U(Z3,t+1) + q_1 q_2 U(Z4,t+1)$  
$U(Z2,t) = δ^t (S, T) + q_2 U(Z4,t+1) + (1-q_2) U(Z6,t+1)$  
$U(Z3,t) = δ^t (T, S) + q_1 U(Z4,t+1) + (1-q_1) U(Z5,t+1)$  
$U(Z4,t) = δ^t (P, P) + U(Z4,t+1)$  
$U(Z5,t) = δ^t (S, T) + (1-q_1) U(Z1,t+1) + q_1 U(Z3,t+1)$  
$U(Z6,t) = δ^t (T, S) + (1-q_2) U(Z1,t+1) + q_2 U(Z2,t+1)$  

Daraus ergeben sich die Bellman-Gleichungen:  
$V_1 = u_1 + δ \cdot ((1-q_1) (1-q_2) V_1 + (1-q_1) q_2 V_2 + q_1 (1-q_2) V_3 + q_1 q_2 V_4)$  
$V_2 = u_2 + δ \cdot (q_2 V_4 + (1-q_2) V_6)$  
$V_3 = u_3 + δ \cdot (q_1 V_4 + (1-q_1) V_5)$  
$V_4 = u_4 + δ \cdot (V_4)$  
$V_5 = u_5 + δ \cdot ((1-q_1) V_1 + q_1 V_3)$  
$V_6 = u_6 + δ \cdot ((1-q_2) V_1 + q_2 V_2)$  

und die Übergangsmatrix $M$ der 6 Zustände:  
$M = \begin{bmatrix}
((1-q_1)(1-q_2)) & ((1-q_1)q_2) & (q_1(1-q_2)) & (q_1 q_2) &       0 &       0 \\
               0 &            0 &            0 &     (q_2) &       0 & (1-q_2) \\
               0 &            0 &            0 &     (q_1) & (1-q_1) &       0 \\
               0 &            0 &            0 &         1 &       0 &       0 \\
         (1-q_1) &            0 &        (q_1) &         0 &       0 &       0 \\
         (1-q_2) &        (q_2) &            0 &         0 &       0 &       0 \\
\end{bmatrix}$

Die Lösung der Bellman-Gleichung  
$V = r + δ \cdot M \cdot V ⇔ (I - δ \cdot M) \cdot V = r$  
ist dann einfach:  
$V = (I - δ \cdot M)^{-1} \cdot r$  

Erläuterungen siehe WiederholtesGefangenendilemma.md, Beispiel 5 (Tit for Tat mit Antesten).
"""
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pathlib
from sympy import *

# Speichert den aktuellen plot als png file
def save_plot_png(pngfile):
    if pngfile != "":
        pathlib.Path(pngfile).unlink(missing_ok=True)
        plt.savefig(pngfile)
        print(f"Plot gespeichert in {pngfile}")

# 2D-Plot der erwarteten Absorptionszeiten τ(Zi) zum stationären Zustand Z4 
def plot2d_tau(tau_zi, label_zi, q, title="", pngfile="", figsize=(7, 7)):
    fig = plt.figure(figsize=figsize)
    plt_q = np.linspace(0.1, 1.0, 100) # Wahrscheinlichkeit q
    plt_tau_zi = []
    plt_label_zi = []
    for i, tau in enumerate(tau_zi):
        plt_tau = plt_q.copy()
        for j in range(len(plt_q)):
            plt_tau[j] = simplify(tau.subs(q,plt_q[j])).evalf()
        values_already_in_list = False
        for j, chk_tau in enumerate(plt_tau_zi):
            if np.allclose(chk_tau, plt_tau):
                values_already_in_list = True
                plt_label_zi[j] += f"\n{label_zi[i]}"
                break
        if not values_already_in_list:
            plt_tau_zi.append(plt_tau)
            plt_label_zi.append(f"{label_zi[i]}")
    for i in range(len(plt_tau_zi)):
        plt.plot(plt_q, plt_tau_zi[i], label=plt_label_zi[i])
    plt.xlabel("q")
    plt.ylabel("τ(Zi)")
    plt.legend()
    plt.title(title)
    save_plot_png(pngfile)

# 3D- und 2D-plot der Nutzenfunktionen U(q,δ)
def plot3d_payoff(U, q, d, title="", pngfile="", figsize=(25, 9)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(131, projection='3d')
    plt_q = np.linspace(0.0, 1.0, 20, endpoint=True) # Wahrscheinlichkeit q
    plt_d = np.linspace(0.0, 1.0, 20, endpoint=False) # Diskontfaktor δ
    plt_q, plt_d = np.meshgrid(plt_q, plt_d) # q,d-grid
    assert(plt_q.shape == plt_d.shape)
    # 3D-plot der Nutzenfunktion U(q,δ)
    plt_U = np.zeros(plt_q.shape)
    for i in range(plt_q.shape[0]):
        for j in range(plt_q.shape[1]):
            plt_U[i][j] = U.subs({q: plt_q[i][j], d: plt_d[i][j]})
    surf = ax.plot_surface(plt_q, plt_d, plt_U, cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5)
    ax.set_xlabel("Wahrscheinlichkeit q = P(Verrat)")
    ax.set_ylabel("Diskontfaktor δ")
    ax.set_zlabel("Nutzen U(q,δ)")
    plt.tight_layout()
    # 2D-plot der Nutzenfunktion U(δ) für 0 <= δ < 0.8 und q = 0, 0.5, 1
    for plt_cnt in [1, 2]:
        ax = fig.add_subplot(132) if plt_cnt == 1 else fig.add_subplot(133)
        d_interval = Interval(0.0, 0.8) if plt_cnt == 1 else Interval(0.8, 1.0)
        bbox = ax.get_position()
        ax.set_position([bbox.bounds[0], 0.1 + bbox.bounds[1], 0.7 * bbox.bounds[2], 0.7 * bbox.bounds[3]])
        plt_ds = np.linspace(d_interval.inf, d_interval.sup, 100, endpoint=False) # Diskontfaktor δ
        for plt_q in [ 0, 0.5, 1 ]: # Wahrscheinlichkeit q
            plt_U = [ U.subs({q: plt_q, d: plt_d}) for plt_d in plt_ds ]
            ax.plot(plt_ds, plt_U, label=f"q={plt_q}")
        plt.xlabel("Diskontfaktor δ")
        plt.ylabel("Nutzen U")
        plt.legend()
    plt.suptitle(f"{title}")
    save_plot_png(pngfile)

# Berechnung der diskontierter Nutzen im unendlich oft wiederholten Gefangenendilemma mit Markov-Prozess 
def prisoners_dilemma_markov():
    # Symbole
    d = symbols('d', real = True, positive=True) # Diskontfaktor δ
    q, q1, q2, d = symbols('q q1 q2 d', real = True, positive=True) # Übergangswahrscheinlichkeiten q1, q2
    R, S, T, P = 3, 1, 4, 2 # symbols('R S T P', real = True) # Nutzen

    # Übergangsmatrix
    M = Matrix([
        [(1-q1)*(1-q2), (1-q1)*q2,     q1*(1-q2), q1*q2, 0,    0],
        [0,              0,            0,         q2,    0, 1-q2],
        [0,              0,            0,         q1,    1-q1, 0],
        [0,              0,            0,         1,     0,    0],
        [1-q1,           0,            q1,        0,     0,    0],
        [1-q2,           q2,           0,         0,     0,    0],
    ])
    # Check: Summe der von einem Knoten ausgehenden Übergangswahrscheinlichkeiten muss gleich 1 sein
    for i in range(6):
        assert(simplify(sum(M[i,:])) == 1) 
    # 6x6 Einheitsmatrix
    I = eye(6)

    # Sofortnutzen Spieler 1 und Spieler 2
    r1 = Matrix([R, S, T, P, S, T]) # Spieler 1
    r2 = Matrix([R, T, S, P, T, S]) # Spieler 2

    # Geschlossene Lösung: $V = (I - δ \cdot P)^{-1} \cdot r$
    V1 = (I - d*M).inv().multiply(r1)
    V2 = (I - d*M).inv().multiply(r2)
    # V1 = (I - d*M).LUsolve(r1) # identisch zu V1 = (I - d*M).inv().multiply(r1)
    # V2 = (I - d*M).LUsolve(r2) # identisch zu V2 = (I - d*M).inv().multiply(r2)

    print(f"V-Matrix Spieler 1: V1 = {V1}\n")
    print(f"V-Matrix Spieler 2: V2 = {V2}\n")
    print(f"Startwert Zustand 1, diskontierter Nutzen für Spieler 1: V1[0] = {V1[0]}\n")
    print(f"Startwert Zustand 1, diskontierter Nutzen für Spieler 2: V2[0] = {V2[0]}\n")

    # Der Vektor x ist der stationäre Zustand, wenn $M^T \cdot x = x$ gilt. x ist die Lösung von $(M^T − I) \cdot x = 0$. 
    # In diesem Beispiel wäre x = [ 0 0 0 1 0 0 ]^T zu erwarten, weil Z4 = (G,G) ein absorbierender Zustand ist.
    M_q = M.subs({q1: q, q2: q}) # Lösung nur für q1 = q2 = q
    stationary = (M_q.T - I).nullspace()
    # Check: stationary = [ 0 0 0 1 0 0 ]. Mit M_q = M_q.subs(q:0) erhält man zusätzlich auch die Lösung [ 1 0 0 0 0 0 ], d.h. für q=0 ist auch Zustand 1 stationär.
    # Sobald q minimal von 0 abweicht, ist aber nur Z4 = (G,G) stationär.

    # Anzahl der zu erwartenden Schritte tau(Zi), die erforderlich sind um von einem Zustand Zi das erste Mal zum stationären Endzustand Z4 zu gelangen
    M_q_tau = M.subs({q1: q, q2: q}).copy()
    M_q_tau.col_del(3) # Z4 (index 3) entfernen
    M_q_tau.row_del(3) # Z4 (index 3) entfernen
    # tau = 1 + M_q_tau * tau
    # <=> (I - M_q_tau) * tau = [1, 1, 1, 1, 1]^T
    # <=> tau = (I - M_q_tau)^{-1} * [1, 1, 1, 1, 1]^T
    ones_5 = Matrix(np.ones(5)) # [1, 1, 1, 1, 1]
    tau = (eye(5) - M_q_tau).inv().multiply(ones_5)

    print(f"Anzahl der zu erwartenden Schritte tau(Zi), um von einem Zustand Zi das erste Mal zum stationären Endzustand Z4 zu gelangen:")
    tau_labels = ["τ(Z1)","τ(Z2)","τ(Z3)","τ(Z5)","τ(Z6)"]
    for i in range(5):
        tau[i] = simplify(tau[i])
        print(f"{tau_labels[i]} = {tau[i]}")
    plot2d_tau(tau, tau_labels, q, title="Erwartete Absorptionszeiten τ(Zi) zum stationären Zustand Z4", pngfile="prisoners_dilemma_markov_tau.png")
    print(f"")

    # Interessante Grenzfälle:
    # q1 = q2 = q
    # q -> 0 (klassisches Tit for Tat)
    # q -> 1 (permanenter Verräter)
    # δ -> 1 (maximale Geduld)
    U1_qd = V1[0].subs({q1: q, q2: q}) # diskontierter Nutzen U1(q,δ) für Spieler 1 bei q1 = q2 = q
    U2_qd = V2[0].subs({q1: q, q2: q}) # diskontierter Nutzen U2(q,δ) für Spieler 1 bei q1 = q2 = q
    print(f"Diskontierter Nutzen für Spieler 1 mit q1=q2=q: U1(q,δ) = {U1_qd}\n")
    print(f"Diskontierter Nutzen für Spieler 2 mit q1=q2=q: U2(q,δ) = {U2_qd}\n")
    plot3d_payoff(U1_qd, q, d, title="Diskontierter Nutzen U1(q,δ) für Spieler 1 mit q1=q2=q", pngfile="prisoners_dilemma_markov_payoff1.png")
    # plot3d_payoff(U2_qd, q, d, title="Diskontierter Nutzen U2(q,δ) für Spieler 2 mit q1=q2=q", pngfile="prisoners_dilemma_markov_payoff2.png")

if __name__ == "__main__":
    prisoners_dilemma_markov()
    plt.show()
    print(f"")
