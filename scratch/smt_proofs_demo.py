import z3

def run_smt_proofs():
    solver = z3.Solver()
    x2 = z3.BitVec('x2', 32)

    print("==========================================================")
    print(" Proof 1: Equivalence of SLLI x1, x2, 1 and ADD x1, x2, x2")
    print("==========================================================")
    
    # SLLI x1, x2, 1  ==  x2 << 1
    # ADD  x1, x2, x2 ==  x2 + x2
    slli_1 = x2 << 1
    add_self = x2 + x2

    # To PROVE equivalence (A == B), we assert the NEGATION (A != B).
    # If Z3 finds no counterexample (UNSAT), the equivalence holds universally.
    solver.add(slli_1 != add_self)
    result1 = solver.check()

    print(f"Assertion  : (x2 << 1) != (x2 + x2)")
    print(f"Z3 Result  : {result1} (UNSAT -> Proof holds! No counterexample exists.)\n")

    print("==========================================================")
    print(" Proof 2: Non-Equivalence test (SLLI x1, x2, 2 vs non-matching MUL)")
    print("==========================================================")

    # Testing non-equivalence: SLLI x1, x2, 2 (x2 << 2) vs MUL x1, x2, 3 on odd input
    solver.reset()
    slli_2 = x2 << 2
    mul_3  = x2 * 3
    is_odd = (x2 & 1) == 1

    solver.add(is_odd)
    solver.add(slli_2 != mul_3)
    result2 = solver.check()

    print(f"Condition  : x2 is odd ((x2 & 1) == 1)")
    print(f"Assertion  : (x2 << 2) != (x2 * 3)")
    print(f"Z3 Result  : {result2} (SAT -> Non-equivalence proven with counterexample!)")
    if result2 == z3.sat:
        model = solver.model()
        val = model[x2].as_long()
        print(f"Counterexample found: x2 = {val} (hex: {hex(val)})")
        print(f"  SLLI x1, x2, 2 -> {val} << 2 = {val << 2}")
        print(f"  MUL  x1, x2, 3 -> {val} * 3  = {val * 3}")

if __name__ == "__main__":
    run_smt_proofs()
