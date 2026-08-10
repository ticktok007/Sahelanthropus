import z3

def test_proofs():
    solver = z3.Solver()
    
    # 32-bit BitVec for x2
    x2 = z3.BitVec('x2', 32)
    
    # Test 1: SLLI x1, x2, 1  vs  ADD x1, x2, x2
    slli_1 = x2 << 1
    add_self = x2 + x2
    
    # Prove equivalence by asserting negation (slli_1 != add_self)
    solver.add(slli_1 != add_self)
    res1 = solver.check()
    print(f"Test 1 (SLLI x1,x2,1 != ADD x1,x2,x2): {res1}")  # Expected UNSAT
    
    # Test 2: Check SLLI x1, x2, 2 vs MUL x1, x2, 4
    solver.reset()
    slli_2 = x2 << 2
    mul_4 = x2 * 4
    odd_cond = (x2 & 1) == 1
    
    # Negation of equivalence on odd input:
    solver.add(odd_cond)
    solver.add(slli_2 != mul_4)
    res2 = solver.check()
    print(f"Test 2 (Odd x2 AND SLLI x1,x2,2 != MUL x1,x2,4): {res2}")
    
    # Test 3: What if we test non-equivalence against a different multiplier (e.g. MUL x1, x2, 3 or SRLI)?
    solver.reset()
    solver.add(odd_cond)
    solver.add(slli_2 != x2 * 3)
    res3 = solver.check()
    print(f"Test 3 (Odd x2 AND SLLI x1,x2,2 != MUL x1,x2,3): {res3}")

if __name__ == "__main__":
    test_proofs()
