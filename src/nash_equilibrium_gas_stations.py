"""
Beispiel für ein Nash-Gleichgewicht: 

Zwei bzw. drei nichtkooperative Tankstellen T[0], T[1], T[2] mit reinen Preisstrategien konkurrieren um Kunden. Alle beziehen ihr Benzin vom gleichen Großhändler 
zu 1 Euro/liter und haben gleiche Fixkosten von 1000 Euro/Tag für Gehälter, Pacht, Steuern, etc. Wir nehmen an, dass die Tankstellen zusammen 20000 liter/Tag 
verkaufen, wenn die Tankstellen ihr Benzin mit 2 Euro/liter anbieten. Also 10000 liter/Tag/Tankstelle bei 2 Tankstellen und 6666.7 liter/Tag/Tankstelle bei 3 Tankstellen
und identischem Verkaufspreis von 2 Euro/liter.

Seien vp[0] der Verkaufspreis in Euro/liter an Tankstelle T[0], x[0] die verkaufte Menge in liter/Tag, ep[0] = 1 der Einkaufspreis in Euro/liter, 
mwst[0] = 0.19 die Umsatzsteuer und fixcost[0] = 1000 die Fixkosten der Tankstelle in Euro/Tag. 
Entsprechend vp[1], x[1], ep[1], mwst[1], fixcost[1] für Tankstelle T[1] und vp[2], x[2], ep[2], mwst[2], fixcost[2] für Tankstelle T[2].

Der Umsatz einer Tankstelle hänge von der Preisdifferenz zum Konkurrenten ab. Je billiger das Benzim im Vergleich zum Konkurrenten wird, desto höher der Verkauf; 
aber nicht schlagartig, sondern stetig und monoton, da die Wartezeit bei größerem Ansturm länger wird. Die Preisdifferenz ist quasi bezahlte Wartezeit der Kunden. 
1 Euro Preisdifferenz verkaufe 5000 liter mehr zu Lasten des Konkurrenten und der Zusammenhang sei der Einfachheit halber linear.
Für 2 Tankstellen:
Δx[0] = 5000 * (vp[1] - vp[0])
Δx[1] = 5000 * (vp[0] - vp[1])
Für 3 Tankstellen:
Δx[0] = 5000 * (vp[2] - vp[0]) + 5000 * (vp[1] - vp[0])
Δx[1] = 5000 * (vp[2] - vp[1]) + 5000 * (vp[0] - vp[1])
Δx[2] = 5000 * (vp[1] - vp[2]) + 5000 * (vp[0] - vp[2])

Gleichzeitig wird insgesamt mehr Benzin verbraucht, wenn der Preis sinkt. Die Nachfrage steige mit sinkendem Verkaufspreis: 
Δx[i] = 1000 * (2 - vp[i])

Für 2 Tankstellen ergibt sich daraus:
x[0] = 20000/2 + 5000 * (vp[1] - vp[0]) + 1000 * (2 - vp[0]) = 5000 * vp[1] - 6000 * vp[0] + 12000
x[1] = 20000/2 + 5000 * (vp[0] - vp[1]) + 1000 * (2 - vp[1]) = 5000 * vp[0] - 6000 * vp[1] + 12000

Für 3 Tankstellen ergibt sich:
x[0] = 20000/3 + 5000 * (vp[2] - vp[0]) + 5000 * (vp[1] - vp[0]) + 1000 * (2 - vp[0]) = 5000 * vp[2] + 5000 * vp[1] - 11000 * vp[0] + 26000/3
x[1] = 20000/3 + 5000 * (vp[2] - vp[1]) + 5000 * (vp[0] - vp[1]) + 1000 * (2 - vp[1]) = 5000 * vp[2] + 5000 * vp[0] - 11000 * vp[1] + 26000/3
x[2] = 20000/3 + 5000 * (vp[1] - vp[2]) + 5000 * (vp[0] - vp[2]) + 1000 * (2 - vp[2]) = 5000 * vp[1] + 5000 * vp[0] - 11000 * vp[2] + 26000/3

Jede Tankstelle T[i] möchte ihren Gewinn g[i] = (vp[i] / (1 + mwst[i]) - ep[i]) * x[i] - fixcost[i] maximieren.

Bei 2 konkurrierenden Tankstellen ergibt sich:
g[0] = (vp[0] / (1 + mwst[0]) - ep[0]) * x[0] - fixcost[0] = (vp[0] / (1 + mwst[0]) - ep[0]) * (5000 * vp[1] - 6000 * vp[0] + 12000) - fixcost[0]
g[0] = (5000/(1+mwst[0]))*vp[1]*vp[0] - 5000*ep[0]*vp[1] - (6000/(1+mwst[0]))*vp[0]*vp[0] + 6000*ep[0]*vp[0] + (12000/(1+mwst[0]))*vp[0] - 12000*ep[0] - fixcost[0]
g[1] = (vp[1] / (1 + mwst[1]) - ep[1]) * x[1] - fixcost[1] = (vp[1] / (1 + mwst[1]) - ep[1]) * (5000 * vp[0] - 6000 * vp[1] + 12000) - fixcost[1]
g[1] = (5000/(1+mwst[1]))*vp[1]*vp[0] - 5000*ep[1]*vp[0] - (6000/(1+mwst[1]))*vp[1]*vp[1] + 6000*ep[1]*vp[1] + (12000/(1+mwst[1]))*vp[1] - 12000*ep[1] - fixcost[1]

Bei 3 konkurrierenden Tankstellen ergibt sich:
g[0] = (vp[0] / (1 + mwst[0]) - ep[0]) * x[0] - fixcost[0] = (vp[0] / (1 + mwst[0]) - ep[0]) * (5000 * vp[2] + 5000 * vp[1] - 11000 * vp[0] + (26000/3)) - fixcost[0]
g[0] = (5000/(1+mwst[0]))*vp[2]*vp[0] + (5000/(1+mwst[0]))*vp[1]*vp[0] - (11000/(1+mwst[0]))*vp[0]*vp[0] - 5000*ep[0]*vp[2] - 5000*ep[0]*vp[1] + ((26000/3)/(1+mwst[0]))*vp[0] + 11000*ep[0]*vp[0] - (26000/3)*ep[0] - fixcost[0]
g[1] = (vp[1] / (1 + mwst[1]) - ep[1]) * x[1] - fixcost[1] = (vp[1] / (1 + mwst[1]) - ep[1]) * (5000 * vp[2] + 5000 * vp[0] - 11000 * vp[1] + (26000/3)) - fixcost[1]
g[1] = (5000/(1+mwst[1]))*vp[2]*vp[1] + (5000/(1+mwst[1]))*vp[0]*vp[1] - (11000/(1+mwst[1]))*vp[1]*vp[1] - 5000*ep[1]*vp[2] - 5000*ep[1]*vp[0] + ((26000/3)/(1+mwst[1]))*vp[1] + 11000*ep[1]*vp[1] - (26000/3)*ep[1] - fixcost[1]
g[2] = (vp[2] / (1 + mwst[2]) - ep[2]) * x[2] - fixcost[2] = (vp[2] / (1 + mwst[2]) - ep[2]) * (5000 * vp[1] + 5000 * vp[0] - 11000 * vp[2] + (26000/3)) - fixcost[2]
g[2] = (5000/(1+mwst[2]))*vp[1]*vp[2] + (5000/(1+mwst[2]))*vp[0]*vp[2] - (11000/(1+mwst[2]))*vp[2]*vp[2] - 5000*ep[2]*vp[1] - 5000*ep[2]*vp[0] + ((26000/3)/(1+mwst[2]))*vp[2] + 11000*ep[2]*vp[2] - (26000/3)*ep[2] - fixcost[2]

Ein Extremwert (Maxima oder Minima) einer Funktionen f(x) wird bei ∂f/∂x = 0 erreicht. 

Die Gewinne bei 2 konkurrierenden Tankstellen werden also maximal bei:
∂g[0]/∂vp[0] = (5000/(1+mwst[0]))*vp[1] - (12000/(1+mwst[0]))*vp[0] + 6000*ep[0] + (12000/(1+mwst[0])) = 0
∂g[1]/∂vp[1] = (5000/(1+mwst[1]))*vp[0] - (12000/(1+mwst[1]))*vp[1] + 6000*ep[1] + (12000/(1+mwst[1])) = 0

Die Gewinne bei 3 konkurrierenden Tankstellen werden maximal bei:
∂g[0]/∂vp[0] = (5000/(1+mwst[0]))*vp[2] + (5000/(1+mwst[0]))*vp[1] - (22000/(1+mwst[0]))*vp[0] + 11000*ep[0] + ((26000/3)/(1+mwst[0])) = 0
∂g[1]/∂vp[1] = (5000/(1+mwst[1]))*vp[2] + (5000/(1+mwst[1]))*vp[0] - (22000/(1+mwst[1]))*vp[1] + 11000*ep[1] + ((26000/3)/(1+mwst[1])) = 0
∂g[2]/∂vp[2] = (5000/(1+mwst[2]))*vp[1] + (5000/(1+mwst[2]))*vp[0] - (22000/(1+mwst[2]))*vp[2] + 11000*ep[2] + ((26000/3)/(1+mwst[2])) = 0

Lösung des Gleichungssystems für 2 Tankstellen:
a*x1 + b*x0 + c = 0  ⇔  x1 = (-b/a)*x0 - c/a  ⇔  x0 = (-a/b)*x1 - c/b
d*x1 + e*x0 + f = 0  ⇔  x1 = (-e/d)*x0 - f/d  ⇔  x0 = (-d/e)*x1 - f/e
(-b/a)*x0 - c/a = (-e/d)*x0 - f/d  ⇔  x0 = (c/a - f/d) / (e/d - b/a)
(-a/b)*x1 - c/b = (-d/e)*x1 - f/e  ⇔  x1 = (c/b - f/e) / (d/e - a/b)

Lösung des Gleichungssystems für 3 Tankstellen:
A[0][0]*x2 + A[0][1]*x1 + A[0][2]*x0 = B[0]
A[1][0]*x2 + A[1][1]*x1 + A[1][2]*x0 = B[1]
A[2][0]*x2 + A[2][1]*x1 + A[2][2]*x0 = B[2]
A * x = B  ⇔  x = np.linalg.solve(A, B)

Das Skript berechnet die Nash-Gleichgewichte für das obige Tankstellen-Beispiel bei unterschiedlichen Einkaufspreisen, Fixkosten und Umsatzsteuern und gibt 
Verkaufspreis, Mengen und Gewinn aus, z.B.:

| Anzahl Tankstellen | Einkaufspreis Euro/Liter | Fixkosten Euro/Tag | Umsatzsteuer | Verkaufspreis Euro/Liter | Gewinn Euro/Tag | Gewinn Euro/Liter | Gesamtmenge Liter/Tag |
| ------------------ | ------------------------ | ------------------ | ------------ | ------------------------ | --------------- | ----------------- | --------------------- |
|          2         |              1           |         1000       |     19%      |          2.73            |     11024       |         1.19      |          18531        |
|          2         |              1           |         1000       |      0%      |          2.57            |     13816       |         1.47      |          18857        |
|          2         |              0           |         1000       |     19%      |          1.71            |     13817       |         1.34      |          20571        |
|          2         |              1           |            0       |     19%      |          2.73            |     12024       |         1.30      |          18531        |
|          3         |              1           |         1000       |     19%      |          1.81            |      2588       |         0.38      |          20561        |
|          3         |              1           |         1000       |      0%      |          1.64            |      3490       |         0.50      |          21083        |
|          3         |              0           |         1000       |     19%      |          0.72            |      3822       |         0.48      |          23833        |
|          3         |              1           |            0       |     19%      |          1.81            |      3588       |         0.52      |          20561        |

Interessant:
* Eine dritte Tankstelle führt zur Verringerung des Preises um 34% und zur Verringerung des Gewinns pro Liter um 68%.
* Eine Erhöhung der Umsatzsteuer von 0 auf 19% erhöht den Preis nur um 6-10%, verringert den Gewinns pro Liter aber um 19-24%.
* Eine Erhöhung der Fixkosten um 1000 Euro/Tag beeinflusst nur den Gewinn, nicht den Verkaufspreis. Das ist nicht erstaunlich, da der Fixkostenanteil 
    bei Ableitung der Gewinnfunktion entfällt und damit das Nash-Gleichgewicht der Preise unabhängig von den Fixkosten ist.

Siehe auch SpieltheorieEinfuehrung.md
"""
import numpy as np

def sprint(arr, decimals):
    """
    String representation of an array with rounded values
    """
    return "[" + ",".join([f"{v:.{decimals}f}" for v in arr]) + "]"

def solve_two_linear_equations(a, b, c, d, e, f):
    """
    Solve 2 linear equations a*x1 + b*x0 + c = 0 and d*x1 + e*x0 + f = 0
    and return x0 = (c/a - f/d) / (e/d - b/a) and x1 = (c/b - f/e) / (d/e - a/b)
    """
    x0 = (c/a - f/d) / (e/d - b/a)
    x1 = (c/b - f/e) / (d/e - a/b)
    return np.array([x0, x1]) # identical to np.linalg.solve(np.array([[a, b], [d, e]]), np.array([-c, -f]))

def calc_nash_equilibrium_for_two_gas_stations(ep = [1.0, 1.0], mwst = [0.19, 0.19], fixcost = [1000.0, 1000.0]):
    """
    Solve the 2 following linear equations
    ∂g[0]/∂vp[0] = (5000/(1+mwst[0]))*vp[1] - (12000/(1+mwst[0]))*vp[0] + 6000*ep[0] + (12000/(1+mwst[0])) = 0
    ∂g[1]/∂vp[1] = (5000/(1+mwst[1]))*vp[0] - (12000/(1+mwst[1]))*vp[1] + 6000*ep[1] + (12000/(1+mwst[1])) = 0
    and return gas prices vp, sales volumes x and profits g
    """
    vp = solve_two_linear_equations(5000/(1+mwst[0]), -12000/(1+mwst[0]), 6000*ep[0] + 12000/(1+mwst[0]), -12000/(1+mwst[1]), 5000/(1+mwst[1]), 6000*ep[1] + 12000/(1+mwst[1]))
    x = np.array([5000 * vp[1] - 6000 * vp[0] + 12000, 5000 * vp[0] - 6000 * vp[1] + 12000])
    margin = vp / (1 + np.array(mwst)) - np.array(ep)
    g = margin * x - np.array(fixcost)
    return vp, x, g

def calc_nash_equilibrium_for_three_gas_stations(ep = [1.0, 1.0, 1.0], mwst = [0.19, 0.19, 0.19], fixcost = [1000.0, 1000.0, 1000.0]):
    """
    Solve the 3 following linear equations
    ∂g[0]/∂vp[0] = (5000/(1+mwst[0]))*vp[2] + (5000/(1+mwst[0]))*vp[1] - (22000/(1+mwst[0]))*vp[0] + 11000*ep[0] + ((26000/3)/(1+mwst[0])) = 0
    ∂g[1]/∂vp[1] = (5000/(1+mwst[1]))*vp[2] + (5000/(1+mwst[1]))*vp[0] - (22000/(1+mwst[1]))*vp[1] + 11000*ep[1] + ((26000/3)/(1+mwst[1])) = 0
    ∂g[2]/∂vp[2] = (5000/(1+mwst[2]))*vp[1] + (5000/(1+mwst[2]))*vp[0] - (22000/(1+mwst[2]))*vp[2] + 11000*ep[2] + ((26000/3)/(1+mwst[2])) = 0
    and return gas prices vp, sales volumes x and profits g
    """
    A = np.array([
        [5000/(1+mwst[0]), 5000/(1+mwst[0]), -22000/(1+mwst[0])], 
        [5000/(1+mwst[1]), -22000/(1+mwst[1]), 5000/(1+mwst[1])],
        [-22000/(1+mwst[2]), 5000/(1+mwst[2]), 5000/(1+mwst[2])] ])
    B = np.array([
        -(11000*ep[0] + (26000/3)/(1+mwst[0])), 
        -(11000*ep[1] + (26000/3)/(1+mwst[1])),
        -(11000*ep[2] + (26000/3)/(1+mwst[2])) ])
    vp = np.linalg.solve(A, B)
    x = np.array([
        5000 * vp[2] + 5000 * vp[1] - 11000 * vp[0] + (26000/3),
        5000 * vp[2] + 5000 * vp[0] - 11000 * vp[1] + (26000/3),
        5000 * vp[1] + 5000 * vp[0] - 11000 * vp[2] + (26000/3) ])
    margin = vp / (1 + np.array(mwst)) - np.array(ep)
    g = margin * x - np.array(fixcost)
    return vp, x, g

def example_nash_equilibrium_gas_stations(N = 2, ep = [1.0, 1.0], mwst = [0.19, 0.19], fixcost = [1000.0, 1000.0]):
    """
    Example of a nash_equilibrium with 2-3 competing and non-cooperative gas stations and pure pricing strategies
    """
    if N == 2: # Calculate the Nash equilibrium for 2 players
        vp, x, g = calc_nash_equilibrium_for_two_gas_stations(ep, mwst, fixcost)
    elif N == 3: # Calculate the Nash equilibrium for 3 players
        vp, x, g = calc_nash_equilibrium_for_three_gas_stations(ep, mwst, fixcost)
    else:
        raise ValueError(f"## ERROR: {N} gas stations NOT supported, use N=2 or N=3")
    print(f"Nash equilibrium for {N} gas stations at purchase prices {sprint(ep,2)} euro/liter, taxes {sprint(mwst,2)}%, fixcost {sprint(fixcost,1)} euro/day:")
    print(f"  salesprice = {sprint(vp,2)} euro/liter, volume = {sprint(x,1)} = {np.sum(x):.1f} liter/day, profit = {sprint(g,2)} euro/day = {sprint(g/x,2)} euro/liter")
    # Simulate the same scenario by iterating over sales prices maximizing the profit for each player, should converge to the Nash equilibrium
    simulation = GasStationSimulation(N = N, ep = ep, mwst = mwst, fixcost = fixcost)
    simulation.simulate(max_iterations = 1000)
    print(f"  Cross-check by simulation: salesprice = {sprint(simulation.vp,2)} euro/liter, volume = {sprint(simulation.x,1)} = {np.sum(simulation.x):.1f} liter/day, profit = {sprint(simulation.g,2)} euro/day = {sprint(simulation.g/simulation.x,2)} euro/liter")

class GasStationSimulation:
    """
    Simulation of competing and non-cooperative gas stations with pure pricing strategies
    """
    def __init__(self, N = 2, ep = [1.0, 1.0], mwst = [0.19, 0.19], fixcost = [1000.0, 1000.0]):
        self.N = N
        self.ep = np.array(ep)
        self.mwst = np.array(mwst)
        self.fixcost = np.array(fixcost)
        self.vp = np.array(ep) # start with sales price = purchase price (or any other reasonable initial price)
        self.x = np.zeros(N, dtype=float)
        self.g = np.zeros(N, dtype=float)
        self.calculate_volume_profits()

    def calculate_volume_profits(self):
        """
        Calculate sales volumes and profits for current sales prices
        """
        if self.N == 2:
            self.x[0] = 5000 * self.vp[1] - 6000 * self.vp[0] + 12000
            self.x[1] = 5000 * self.vp[0] - 6000 * self.vp[1] + 12000
        elif self.N == 3:
            self.x[0] = 5000 * self.vp[2] + 5000 * self.vp[1] - 11000 * self.vp[0] + 26000/3
            self.x[1] = 5000 * self.vp[2] + 5000 * self.vp[0] - 11000 * self.vp[1] + 26000/3
            self.x[2] = 5000 * self.vp[1] + 5000 * self.vp[0] - 11000 * self.vp[2] + 26000/3
        else:
            raise ValueError(f"## ERROR in GasStationSimulation: {N} gas stations NOT supported, use N=2 or N=3")
        margin = self.vp / (1 + self.mwst) - self.ep
        self.g = margin * self.x - self.fixcost

    def simulate(self, max_iterations = 1000):
        """
        Simulate competing gas stations adjusting their sales prices to maximize their profits, should converge to the Nash equilibrium
        """
        for iteration in range(max_iterations):
            vp_modified = False
            for n in range(self.N):
                current_g = self.g[n]
                current_vp = self.vp[n]
                # increase sales price by 1 cent and check if profit increases
                self.vp[n] = current_vp + 0.01
                self.calculate_volume_profits()
                if self.g[n] >= current_g:
                    vp_modified = True
                    continue
                # decrease sales price by 1 cent and check if profit increases
                self.vp[n] = current_vp - 0.01
                self.calculate_volume_profits()
                if self.g[n] > current_g:
                    vp_modified = True
                    continue
                # restore previous sales price if profit did not increase
                self.vp[n] = current_vp
                self.calculate_volume_profits()
            if not vp_modified:
                break

if __name__ == "__main__":
    """
    Examples of competing and non-cooperative gas stations and pure pricing strategies
    """

    example_nash_equilibrium_gas_stations(N = 2, ep = [1.0, 1.0], mwst = [0.19, 0.19], fixcost = [1000.0, 1000.0]) # reference scenario with purchase prices, taxes, fixcosts and 2 gas stations
    example_nash_equilibrium_gas_stations(N = 2, ep = [1.0, 1.0], mwst = [0.00, 0.00], fixcost = [1000.0, 1000.0]) # zero taxes
    example_nash_equilibrium_gas_stations(N = 2, ep = [0.0, 0.0], mwst = [0.19, 0.19], fixcost = [1000.0, 1000.0]) # zero purchase prices
    example_nash_equilibrium_gas_stations(N = 2, ep = [1.0, 1.0], mwst = [0.19, 0.19], fixcost = [   0.0,    0.0]) # zero fixcosts

    example_nash_equilibrium_gas_stations(N = 3, ep = [1.0, 1.0, 1.0], mwst = [0.19, 0.19, 0.19], fixcost = [1000.0, 1000.0, 1000.0]) # reference scenario with purchase prices, taxes, fixcosts and 3 gas stations
    example_nash_equilibrium_gas_stations(N = 3, ep = [1.0, 1.0, 1.0], mwst = [0.00, 0.00, 0.00], fixcost = [1000.0, 1000.0, 1000.0]) # zero taxes
    example_nash_equilibrium_gas_stations(N = 3, ep = [0.0, 0.0, 0.0], mwst = [0.19, 0.19, 0.19], fixcost = [1000.0, 1000.0, 1000.0]) # zero purchase prices
    example_nash_equilibrium_gas_stations(N = 3, ep = [1.0, 1.0, 1.0], mwst = [0.19, 0.19, 0.19], fixcost = [   0.0,    0.0,    0.0]) # zero fixcosts
