"""
Gefangenendilemma: Ein (und nur ein) Gefangener kann die Rolle als Kronzeuge wählen, d.h. seine Tat gestehen, die anderen Gefangenen belasten und dafür Strafminderung erhalten. 
Leugnen beide Gefangene, erhält keiner Strafminderung (Nutzen = -2). Gesteht einer, erhält dieser Strafminderung (Nutzen = -1), der andere Strafverschärfung (Nutzen = -10). 
Gestehen beide, erhalten beide längere Strafen (Nutzen = -5). 2x2-Matrix des Gefangenendilemma:

    |           | **Gefangener 2 wählt Leugnen** | **Gefangener 2 wählt Gestehen** |   | **$\min(u_1)$** |
    | ------------------------------- | -------- | -------- |---| ----------------- |
    | **Gefangener 1 wählt Leugnen**  | -2 ;  -2 | -10 ; -1 |   | $\min(u_1,s_{11}) = -10$ |
    | **Gefangener 1 wählt Gestehen** | -1 ; -10 |  -5 ; -5 |   | $\min(u_1,s_{12}) =  -5$ |
    | **$\min(u_2)$**                 | $\min(u_2,s_{21}) = -10$ | $\min(u_2,s_{22}) = -5$ |   |   |

    Für beide Gefangenen ist Strategie 2 (Gestehen) optimal mit Nutzen von -5 im ungünstigsten Fall.

Dieses skript löst das Gefangenendilemma unter Verwendung der python libraries nashpy und pygambit
* nashpy install: python -m pip install nashpy
* nashpy tutorial: https://nashpy.readthedocs.io/en/stable/how-to/calculate-utilities.html
* pygambit install: python -m pip install pygambit
* pygambit tutorial: https://gambitproject.readthedocs.io/en/stable/tutorials/01_quickstart.html

Beispielausgabe:
    Prisoners dilemma (nashpy):
        payoff matrix player A: [[ -2,-10], [ -1, -5]]
        payoff matrix player B: [[ -2, -1], [-10, -5]]
        best response for player B: Testify
        best response for player A: Testify
        best response for player A: Testify
        best response for player B: Testify
        Nash equilibrium: player A: Testify, player B: Testify with payoffs [-5 -5]
    Prisoners dilemma (pygambit):
        payoff matrix player A: [[-2,-10], [-1,-5]]
        payoff matrix player B: [[-2,-1], [-10,-5]]
        Nash equilibrium: player A: Testify (p=1, payoff=-5)
        Nash equilibrium: player B: Testify (p=1, payoff=-5)

Siehe auch https://de.wikipedia.org/wiki/Gefangenendilemma und SpieltheorieEinfuehrung.md
"""
import numpy as np
import nashpy as nash
import pygambit as gbt

def matrix2oneline(m, sep=","):
    # Shortcut to print a matrix in one line with a given separator
    return np.array2string(m,separator=sep).replace('\n','')

def nashpy_prisoners_dilemma():
    # Solve the prisoners dilemma using nashpy
    payoff_player_A = np.array([[-2, -10], [ -1, -5]]) # payoff_player_A = np.array([[3, 0], [5, 1]])
    payoff_player_B = np.array([[-2,  -1], [-10, -5]]) # payoff_player_B = np.array([[3, 5], [0, 1]])
    prisoners_dilemma = nash.Game(payoff_player_A, payoff_player_B)
    print(f"Prisoners dilemma (nashpy):")
    print(f"    payoff matrix player A: {matrix2oneline(prisoners_dilemma.payoff_matrices[0])}")
    print(f"    payoff matrix player B: {matrix2oneline(prisoners_dilemma.payoff_matrices[1])}")
    strategy_names = ["Silent", "Testify"]
    for strategy_idx_player_A, strategy_arr_player_A in enumerate([[1, 0], [0, 1]]):
        for strategy_idx_player_B, strategy_arr_player_B in enumerate([[1, 0], [0, 1]]):
            best_response = prisoners_dilemma.is_best_response(np.array(strategy_arr_player_A), np.array(strategy_arr_player_B))
            if best_response[0]: # best response for player A
                print(f"    best response for player A: {strategy_names[strategy_idx_player_A]}")
            if best_response[1]: # best response for player B
                print(f"    best response for player B: {strategy_names[strategy_idx_player_B]}")
            if best_response[0] and best_response[1]: # best responses for both player A and B found
                print(f"    Nash equilibrium: player A: {strategy_names[strategy_idx_player_A]}, player B: {strategy_names[strategy_idx_player_B]} with payoffs {prisoners_dilemma[strategy_arr_player_A, strategy_arr_player_B]}")
    print()

def gambit_prisoners_dilemma():
    # Solve the prisoners dilemma using pygambit
    payoff_player_A = np.array([[-2, -10], [ -1, -5]]) # payoff_player_A = np.array([[3, 0], [5, 1]])
    payoff_player_B = np.array([[-2,  -1], [-10, -5]]) # payoff_player_B = np.array([[3, 5], [0, 1]])
    game = gbt.Game.from_arrays(payoff_player_A, payoff_player_B, title="Prisoners dilemma")
    gbt_payoffs_A, gbt_payoffs_B = game.to_arrays(dtype=int)
    nash_result = gbt.nash.enumpure_solve(game)
    nash_equilibria = nash_result.equilibria
    print(f"Prisoners dilemma (pygambit):")
    print(f"    payoff matrix player A: {matrix2oneline(gbt_payoffs_A)}")
    print(f"    payoff matrix player B: {matrix2oneline(gbt_payoffs_B)}")
    for equilibrium in nash_equilibria:
        prob_A = [ equilibrium[game.players[0].label][game.strategies[0].label], equilibrium[game.players[0].label][game.strategies[1].label] ] # probability player A strategy 0 (silent) and strategy 1 (testify)
        prob_B = [ equilibrium[game.players[1].label][game.strategies[0].label], equilibrium[game.players[1].label][game.strategies[1].label] ] # probability player B strategy 0 (silent) and strategy 1 (testify)
        if prob_A[0] > prob_A[1]:
            print(f"    Nash equilibrium: player A: Silent (p={prob_A[0]}, payoff={equilibrium.payoff(game.players[0].label)})")
        elif prob_A[0] < prob_A[1]:
            print(f"    Nash equilibrium: player A: Testify (p={prob_A[1]}, payoff={equilibrium.payoff(game.players[0].label)})")
        else:
            print(f"    Nash equilibrium: player A: Silent or Testify (p={prob_A[0]}={prob_A[1]}, payoff={equilibrium.payoff(game.players[0].label)})")
        if prob_B[0] > prob_B[1]:
            print(f"    Nash equilibrium: player B: Silent (p={prob_B[0]}, payoff={equilibrium.payoff(game.players[1].label)})")
        elif prob_B[0] < prob_B[1]:
            print(f"    Nash equilibrium: player B: Testify (p={prob_B[1]}, payoff={equilibrium.payoff(game.players[1].label)})")
        else:
            print(f"    Nash equilibrium: player B: Silent or Testify (p={prob_B[0]}={prob_B[1]}, payoff={equilibrium.payoff(game.players[1].label)})")
    print()

if __name__ == "__main__":
    # Solve the prisoners dilemma using nashpy and pygambit
    nashpy_prisoners_dilemma()
    gambit_prisoners_dilemma()
