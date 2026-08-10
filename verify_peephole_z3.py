import z3

def run_z3_verification():
    print("======================================================================")
    print(" Z3 Equivalence Verification: (MUL by 4  vs  SLLI by 2)")
    print("======================================================================")

    # 32-bit symbolic variable for register t0
    t0 = z3.BitVec('t0', 32)
    
    # Program A (Unoptimized): t1 = 4, t2 = t0 * t1
    val_A = t0 * 4
    
    # Program B (Optimized): t2 = t0 << 2
    val_B = t0 << 2

    solver = z3.Solver()
    
    # Assert negation: Does there exist ANY t0 where (t0 * 4) != (t0 << 2)?
    solver.add(val_A != val_B)
    result = solver.check()

    print(f"Program A Expression : t0 * 4")
    print(f"Program B Expression : t0 << 2")
    print(f"Z3 Negation Query    : (t0 * 4) != (t0 << 2)")
    print(f"Z3 Solver Result     : {result}")
    
    assert result == z3.unsat, "Expected UNSAT for equivalent peephole rewrite!"
    print("======================================================================")
    print("[+] Z3 VERIFICATION CONFIRMED: UNSAT (Equivalence Holds Universally!)")
    print("======================================================================\n")

if __name__ == "__main__":
    run_z3_verification()
