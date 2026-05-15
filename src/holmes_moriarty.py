"""
Sherlock-Holmes wird von Professor Moriarty verfolgt. 
Ein Nullsummenspiel mit folgender 2x2-Spielmatrix:

|                                         | Holmes steigt in Dover aus (q) | Holmes steigt in Canterbury aus (1-q) |
| --------------------------------------- | ------------------------------ | ------------------------------------- |
| Moriarty steigt in Dover aus (p)        |      (a ; -a) = (+1 ; -1)      |         (b ; -b) = ( 0 ;  0)          |
| Moriarty steigt in Canterbury aus (1-p) |      (c ; -c) = (-1 ; +1)      |         (d ; -d) = (+1 ; -1)          |

Allgemeine 2x2-Spielmatrix:

|           | s21 (q) | s22 (1-q) |
| --------- | ------- | --------- |
| s11 (p)   |  a ; b  |   c ; d   |
| s12 (1-p) |  e ; f  |   g ; h   |

Siehe SpieltheorieEinfuehrung.md
"""
from sympy import * # pip install sympy

if __name__ == "__main__":

    # 2x2-Nullsummenspiel (Sherlock-Holmes gegen Professor Moriarty):
    # Erwartungsnutzen für Moriarty:     $E[U_{1}(s_{11})] = a q + b (1-q)$ und $E[U_{1}(s_{12})] = c q + d (1-q)$
    # Indifferenzbedingung für Moriarty: $E[U_{1}(s_{11})] = E[U_{1}(s_{12})] ⇔ a q + b (1-q) = c q + d (1-q)$
    # Erwartungsnutzen für Holmes:       $E[U_{2}(s_{21})] = -a p - c (1-p)$ und $E[U_{2}(s_{22})] = -b p - d (1-p)$
    # Indifferenzbedingung für Holmes:   $E[U_{2}(s_{21})] = E[U_{2}(s_{22})] ⇔ -a p - c (1-p) = -b p - d (1-p)$
    a, b, c, d, p, q = symbols("a b c d p q", real = True)
    sol = solve([Eq(a*q + b*(1-q), c*q + d*(1-q)), Eq(-a*p - c*(1-p), -b*p - d*(1-p))], [p, q])
    print(f"2x2-Nullsummenspiel: {sol}")

    # Beispiel 2x2-Nullsummenspiel mit a = +1, b = 0, c = -1, d = +1 (Sherlock-Holmes gegen Professor Moriarty):
    example_subs = {a:1, b:0, c:-1, d:1}
    print(f"Beispiel Sherlock-Holmes gegen Professor Moriarty: p = {sol[p].subs(example_subs)}, q = {sol[q].subs(example_subs)}")

    # Allgemeine 2x2-Spielmatrix:
    # Erwartungsnutzen für Spieler 1:     $E[U_{1}(s_{11})] = a q + c (1-q)$ und $E[U_{1}(s_{12})] = e q + g (1-q)$
    # Indifferenzbedingung für Spieler 1: $E[U_{1}(s_{11})] = E[U_{1}(s_{12})] ⇔ a q + c (1-q) = e q + g (1-q)$
    # Erwartungsnutzen für Spieler 2:     $E[U_{2}(s_{21})] = b p + f (1-p)$ und $E[U_{2}(s_{22})] = d p + h (1-p)$
    # Indifferenzbedingung für Spieler 2: $E[U_{2}(s_{21})] = E[U_{2}(s_{22})] ⇔ b p + f (1-p) = d p + h (1-p)$
    a, b, c, d, e, f, g, h, p, q = symbols("a b c d e f g h p q", real = True)
    sol = solve([Eq(a*q + c*(1-q), e*q + g*(1-q)), Eq(b*p + f*(1-p), d*p + h*(1-p))], [p, q])
    print(f"2x2-Spielmatrix:     {sol}")

    # Beispiel 2x2-Spielmatrix mit a = +1, b = -1, c = 0, d = 0, e = -1, f = +1, g = +1, h = -1 (Sherlock-Holmes gegen Professor Moriarty):
    example_subs = {a:1, b:-1, c:0, d:0, e:-1, f:1, g:1, h:-1}
    print(f"Beispiel Sherlock-Holmes gegen Professor Moriarty: p = {sol[p].subs(example_subs)}, q = {sol[q].subs(example_subs)}")
