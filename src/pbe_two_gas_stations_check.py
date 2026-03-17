"""
Utilities, supporting functions and PBE checks for pbe_two_gas_stations.py
"""
import numpy as np
import time
from sympy import *
from sympy.assumptions.relation.binrel import AppliedBinaryRelation

# Return true, if var is a symbolic probability
def is_probability(var):
    return isinstance(var, Symbol) and (var.name.startswith("P(") or var.name == "q")

# Replace (Piecewise((expr1,cond),(expr2,True)) <= rhs) by ((cond & (expr1<=rhs)) | (~cond & (expr2<=rhs)))
def replace_piecewise(atom):
    if isinstance(atom.lhs, Piecewise) and len(atom.lhs.args) > 0 and atom.lhs.args[-1].cond == True:
        rhs = atom.rhs
        cond = False
        replaced = False
        for arg in atom.lhs.args:
            cond = Not(cond) & arg.cond
            if (isinstance(atom,Rel) and atom.rel_op == Ge.rel_op) or (isinstance(atom,AppliedBinaryRelation) and atom.function.name.casefold() == "ge"):
                replaced = Or(replaced, cond & Ge(arg.expr, rhs))
            elif (isinstance(atom,Rel) and atom.rel_op == Gt.rel_op) or (isinstance(atom,AppliedBinaryRelation) and atom.function.name.casefold() == "gt"):
                replaced = Or(replaced, cond & Gt(arg.expr, rhs))
            elif (isinstance(atom,Rel) and atom.rel_op == Le.rel_op) or (isinstance(atom,AppliedBinaryRelation) and atom.function.name.casefold() == "le"):
                replaced = Or(replaced, cond & Le(arg.expr, rhs))
            elif (isinstance(atom,Rel) and atom.rel_op == Lt.rel_op) or (isinstance(atom,AppliedBinaryRelation) and atom.function.name.casefold() == "lt"):
                replaced = Or(replaced, cond & Lt(arg.expr, rhs))
            elif (isinstance(atom,Rel) and atom.rel_op == Eq.rel_op) or (isinstance(atom,AppliedBinaryRelation) and atom.function.name.casefold() == "eq"):
                replaced = Or(replaced, cond & Eq(arg.expr, rhs))
            else:
                return atom
        return replaced
    return atom

# Reduce the intervals of AND-combined terms
def reduce_intervals(terms, assume_probabilities_in_range_0_1 = True):
    # Intersect upper and lower limits of ge, ge, le, lt and eq relations of all terms
    terms_reduced = []
    intervals = {}
    for term in terms:
        interval = Interval(-oo, oo)
        if (isinstance(term,Rel) and term.rel_op == Ge.rel_op) or (isinstance(term,AppliedBinaryRelation) and term.function.name.casefold() == "ge"):
            interval = Interval(term.rhs, oo)
        elif (isinstance(term,Rel) and term.rel_op == Gt.rel_op) or (isinstance(term,AppliedBinaryRelation) and term.function.name.casefold() == "gt"):
            interval = Interval.open(term.rhs, oo)
        elif (isinstance(term,Rel) and term.rel_op == Le.rel_op) or (isinstance(term,AppliedBinaryRelation) and term.function.name.casefold() == "le"):
            interval = Interval(-oo, term.rhs)
        elif (isinstance(term,Rel) and term.rel_op == Lt.rel_op) or (isinstance(term,AppliedBinaryRelation) and term.function.name.casefold() == "lt"):
            interval = Interval.open(-oo, term.rhs)
        elif (isinstance(term,Rel) and term.rel_op == Eq.rel_op) or (isinstance(term,AppliedBinaryRelation) and term.function.name.casefold() == "eq"):
            interval = Interval(term.rhs, term.rhs)
        else:
            terms_reduced.append(term)
        if term.lhs in intervals:
            intervals[term.lhs] = intervals[term.lhs].intersect(interval)
        else:
            intervals[term.lhs] = interval
    # Check for tautologies and not satisfiable terms and append all other terms
    for var, interval in intervals.items():
        if interval.is_empty or interval.inf > interval.sup:
            return False # var is not satisfiable, thus the AND combination is not satisfiable
        append_interval_inf = (interval.inf > -oo)
        append_interval_sup = (interval.sup < oo)
        if assume_probabilities_in_range_0_1 and is_probability(var):
            if (interval.is_open and interval.inf < 0) or (not interval.is_open and interval.inf <= 0):
                append_interval_inf = False # ignore terms of type P(x) >= 0
            if (interval.is_open and interval.sup > 1) or (not interval.is_open and interval.sup >= 1):
                append_interval_sup = False # ignore terms of type P(x) <= 1
        if interval.is_open:
            if append_interval_inf:
                terms_reduced.append(Gt(var, interval.inf))
            if append_interval_sup:
                terms_reduced.append(Lt(var, interval.sup))
        elif interval.inf == interval.sup:
            terms_reduced.append(Eq(var, interval.inf))
        else:
            if append_interval_inf:
                terms_reduced.append(Ge(var, interval.inf))
            if append_interval_sup:
                terms_reduced.append(Le(var, interval.sup))
    and_reduced = True
    for term in terms_reduced:
        if term == True:
            continue # always True is unnecessary in a set of AND expressions
        if term == False:
            return False # if one term is not satisfiable, the AND combination is not satisfiable
        and_reduced = And(and_reduced, term)
    and_reduced_simple = simplify(and_reduced)
    # print(f"reduce_intervals({terms}):\n    {and_reduced}\n    {and_reduced_simple}")
    return and_reduced_simple

# Check, ob die Wahrscheinlichkeitswerte aus einem grid eine gegebene PBE-Konsistenzbedingung erfüllen. 
# Liefert pbe_grid_check True, ist die Konsistenz bestätigt. Liefert pbe_grid_check False, wurde keine Belegung 
# der Konsistenzbedingung gefunden (die Konsistenzbedingung könnte aber dennoch wahr sein)
def pbe_grid_check(pbe_condition, symbols_list, grid_step=0.01, timeout_sec=5):
    q, PaMR, PaMA, PeKR, PeKA, PKRe, PMRa, PMRf, PhKR, PhKA = symbols_list
    prob_grid = np.array([[0.5+delta, 0.5-delta] for delta in np.arange(0, 0.5+grid_step, grid_step)]).flatten()[1:] # prob_grid = [0.5, 0.51, 0.49, 0.52, 0.48, ..., 1.0, 0.0]
    has_PaMA = pbe_condition.has(PaMA)
    has_PaMR = pbe_condition.has(PaMR)
    has_PKRe = pbe_condition.has(PKRe)
    has_PMRa = pbe_condition.has(PMRa)
    has_PMRf = pbe_condition.has(PMRf)
    has_PeKR = pbe_condition.has(PeKR)
    has_PeKA = pbe_condition.has(PeKA)
    has_PhKR = pbe_condition.has(PhKR)
    has_PhKA = pbe_condition.has(PhKA)
    start_time = time.time()
    for PaMA_val in prob_grid:
        pbe_cond1 = pbe_condition.subs(PaMA, PaMA_val)
        if not satisfiable(pbe_cond1) == False:
            for PaMR_val in prob_grid:
                pbe_cond2 = pbe_cond1.subs(PaMR, PaMR_val)
                if not satisfiable(pbe_cond2) == False:
                    for PKRe_val in prob_grid:
                        pbe_cond3 = pbe_cond2.subs(PKRe, PKRe_val)
                        if not satisfiable(pbe_cond3) == False:
                            for PMRa_val in prob_grid:
                                pbe_cond4 = pbe_cond3.subs(PMRa, PMRa_val)
                                if not satisfiable(pbe_cond4) == False:
                                    for PMRf_val in prob_grid:
                                        pbe_cond5 = pbe_cond4.subs(PMRf, PMRf_val)
                                        if not satisfiable(pbe_cond5) == False:
                                            for PeKR_val in prob_grid:
                                                pbe_cond6 = pbe_cond5.subs(PeKR, PeKR_val)
                                                if not satisfiable(pbe_cond6) == False:
                                                    for PeKA_val in prob_grid:
                                                        pbe_cond7 = pbe_cond6.subs(PeKA, PeKA_val)
                                                        if not satisfiable(pbe_cond7) == False:
                                                            for PhKR_val in prob_grid:
                                                                pbe_cond8 = pbe_cond7.subs(PhKR, PhKR_val)
                                                                if not satisfiable(pbe_cond8) == False:
                                                                    for PhKA_val in prob_grid:
                                                                        pbe_cond9 = pbe_cond8.subs(PhKA, PhKA_val)
                                                                        if not satisfiable(pbe_cond9) == False:
                                                                            pbe_satisfiable = satisfiable(pbe_cond9)
                                                                            if not pbe_satisfiable == False:
                                                                                str_PaMA_val = f", P(a|M=A)={PaMA_val:.2f}" if has_PaMA else "" 
                                                                                str_PaMR_val = f", P(a|M=R)={PaMR_val:.2f}" if has_PaMR else "" 
                                                                                str_PKRe_val = f", P(K=R|e)={PKRe_val:.2f}" if has_PKRe else "" 
                                                                                str_PMRa_val = f", P(M=R|a)={PMRa_val:.2f}" if has_PMRa else "" 
                                                                                str_PMRf_val = f", P(M=R|f)={PMRf_val:.2f}" if has_PMRf else "" 
                                                                                str_PeKR_val = f", P(e|K=R)={PeKR_val:.2f}" if has_PeKR else "" 
                                                                                str_PeKA_val = f", P(e|K=A)={PeKA_val:.2f}" if has_PeKA else "" 
                                                                                str_PhKR_val = f", P(h|K=R)={PhKR_val:.2f}" if has_PhKR else "" 
                                                                                str_PhKA_val = f", P(h|K=A)={PhKA_val:.2f}" if has_PhKA else "" 
                                                                                grid_check_witness = True
                                                                                grid_check_witness = And(Eq(PaMA, PaMA_val), grid_check_witness) if has_PaMA else grid_check_witness
                                                                                grid_check_witness = And(Eq(PaMR, PaMR_val), grid_check_witness) if has_PaMR else grid_check_witness
                                                                                grid_check_witness = And(Eq(PKRe, PKRe_val), grid_check_witness) if has_PKRe else grid_check_witness
                                                                                grid_check_witness = And(Eq(PMRa, PMRa_val), grid_check_witness) if has_PMRa else grid_check_witness
                                                                                grid_check_witness = And(Eq(PMRf, PMRf_val), grid_check_witness) if has_PMRf else grid_check_witness
                                                                                grid_check_witness = And(Eq(PeKR, PeKR_val), grid_check_witness) if has_PeKR else grid_check_witness
                                                                                grid_check_witness = And(Eq(PeKA, PeKA_val), grid_check_witness) if has_PeKA else grid_check_witness
                                                                                grid_check_witness = And(Eq(PhKR, PhKR_val), grid_check_witness) if has_PhKR else grid_check_witness
                                                                                grid_check_witness = And(Eq(PhKA, PhKA_val), grid_check_witness) if has_PhKA else grid_check_witness
                                                                                if pbe_cond9.has(q):
                                                                                    try:
                                                                                        q_solutions = solve(pbe_cond9.args, q)
                                                                                        if not isinstance(q_solutions, list) or len(q_solutions) == 0:
                                                                                            q_solutions = prob_grid
                                                                                    except NotImplementedError as exc:
                                                                                        q_solutions = prob_grid
                                                                                    for q_val in q_solutions:
                                                                                        pbe_condq = pbe_cond9.subs(q, q_val)
                                                                                        q_satisfiable = satisfiable(pbe_condq)
                                                                                        if not q_satisfiable == False:
                                                                                            str_q_val = f", q={q_val:.2f}"
                                                                                            grid_check_witness = And(pbe_condq, Eq(q, q_val), grid_check_witness)
                                                                                            return True, grid_check_witness, f"PBE grid check bestanden, erfuellbar z.B. durch {pbe_condq}{str_q_val}{str_PaMA_val}{str_PaMR_val}{str_PKRe_val}{str_PMRa_val}{str_PMRf_val}{str_PeKR_val}{str_PeKA_val}{str_PhKR_val}{str_PhKA_val}"
                                                                                else:
                                                                                    grid_check_witness = And(pbe_cond9, grid_check_witness)
                                                                                    grid_check_witness = And(pbe_cond9, Eq(PaMA, PaMA_val), Eq(PaMR, PaMR_val), Eq(PKRe, PKRe_val), Eq(PMRa, PMRa_val), Eq(PMRf, PMRf_val), Eq(PeKR, PeKR_val), Eq(PeKA, PeKA_val), Eq(PhKR, PhKR_val), Eq(PhKA, PhKA_val))
                                                                                    return True,  grid_check_witness, f"PBE grid check bestanden, erfuellbar z.B. durch {pbe_cond9}{str_PaMA_val}{str_PaMR_val}{str_PKRe_val}{str_PMRa_val}{str_PMRf_val}{str_PeKR_val}{str_PeKA_val}{str_PhKR_val}{str_PhKA_val}"
                                                                        if not has_PhKA or time.time() - start_time > timeout_sec:
                                                                            break
                                                                if not has_PhKR or time.time() - start_time > timeout_sec:
                                                                    break
                                                        if not has_PeKA or time.time() - start_time > timeout_sec:
                                                            break
                                                if not has_PeKR or time.time() - start_time > timeout_sec:
                                                    break
                                        if not has_PMRf or time.time() - start_time > timeout_sec:
                                            break
                                if not has_PMRa or time.time() - start_time > timeout_sec:
                                    break
                        if not has_PKRe or time.time() - start_time > timeout_sec:
                            break
                if not has_PaMR or time.time() - start_time > timeout_sec:
                    break
        if not has_PaMA or time.time() - start_time > timeout_sec:
            break
    if time.time() - start_time > timeout_sec:
        return False, False, f"PBE plausibility grid check timeout"
    return False,  False, f"PBE plausibility grid check nicht bestanden"
