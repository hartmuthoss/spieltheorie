# Notizen zur Spieltheorie

Notizen, Gedanken und Skripte rund um die Spieltheorie, auf Einsteigerniveau, von einem Amateur und im Aufbau begriffen.

Inhalt:

* [Einführung in die Spieltheorie](docs/SpieltheorieEinfuehrung.md): Einige Definitionen, Sätze und Beispiele.

* [prisoners_dilemma.py](src/prisoners_dilemma.py): Das klassische Gefangenendilemma unter Verwendung von nashpy und pygambit.

* [nash_equilibrium_gas_stations.py](src/nash_equilibrium_gas_stations.py): Kleines Pythonskript zur Berechnung des Nash-Gleichgewichtes am Beispiel von konkurrierenden, nicht-kooperativen Tankstellen mit reinen Preisstrategien.

* [three_connect.py](src/three_connect.py) und [four_connect.cpp](src/four_connect.cpp): Backtracking am Beispiel "Drei gewinnt" und "Vier gewinnt".

* [pbe_two_gas_stations.py](src/pbe_two_gas_stations.py): Pythonskript zur Berechnung perfekter Bayes-Gleichgewichte (Perfect Bayesian equilibrium, PBE) am Beispiel zweier konkurrierenden Tankstellen.

* Was passiert, wenn abzählbar unendlich viele Käufer, abzählbar unendlich viele Anbieter, eine unendliche Nachfrage und ein unendliches Angebot zusammentreffen? Dieses Problem wird in den [Hilbert-Tankstellen](docs/HilbertTankstellen.md) berechnet.

* Das [unendlich oft wiederholte Gefangenendilemma](docs/WiederholtesGefangenendilemma.md) zeigt Beispiele für das Folk-Theorem, das One-Shot-Deviation-Principle und Markov-Prozesse im unendlich oft wiederholten Spiel.

* Ich habe Wirtschaft nie verstanden. Banken erzeugen Geld scheinbar aus dem Nichts, Vermögen und Schulden wachsen als gäbe es ein perpetuum mobile - wie kann das sein? Kann Geld auf Dauer exponentiell wachsen?  
    Ich versuche ein minimalistisches Wirtschaftsmodell (Mini-WM) zu bauen, das so etwas Ähnliches wie "Wirtschaft" und "Wachstum" aus einfachen Grundannahmen ableiten kann. Das Modell soll nicht möglichst realistisch, sondern möglichst kausal sein.
    * [Teil 1:](docs/MiniWM_Teil01_RobinsonCrusoe.md) Robinson sitzt auf einer einsamen Insel, fischt und sammelt Kokosnüsse. Durch Investitionen und Forschung kann er seine Produktion steigern.
    * [Teil 2:](docs/MiniWM_Teil02_RobinsonFreitag.md) Robinson bekommt Gesellschaft von Freitag. Dank Spieltheorie maximieren sie ihren Nutzen. Warentausch ist kein Nullsummenspiel; Märkte können Pareto-effizient sein oder nicht. Unter bestimmten Umständen kann die gemeinsame Nutzung des Inselwaldes zum "race to exploit" werden.
    * Teil 3: Robinson macht Schulden.
