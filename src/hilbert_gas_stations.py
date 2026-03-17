"""
Hilbert-Tankstellen:
Beschreibung und Erlaeuterungen in ../docs/HilbertTankstellen.md
"""
import math
from enum import IntEnum
from sympy import *
from hilbert_gas_stations_nash import *
from hilbert_gas_stations_two_players import *
from hilbert_gas_stations_equilibrium import *

if __name__ == "__main__":

    nash_equilibrium_hilbert() # Nash-Gleichgewicht an Hilbert-Tankstellen
    equilibrium_two_players()  # Gleichgewichte im 2-Konkurrenten-Spiel
    equilibrium_N_player()     # Gleichgewichte mit N Spielern und N -> oo
