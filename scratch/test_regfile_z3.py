import z3

def test_regfile_symbolic():
    BV5  = z3.BitVecSort(5)
    BV32 = z3.BitVecSort(32)
    BV8  = z3.BitVecSort(8)

    # Initial register file and memory state
    rf = z3.Array('rf', BV5, BV32)
    mem = z3.Array('mem', BV32, BV8)

    # Invariant: x0 is always 0
    ZERO5 = z3.BitVecVal(0, 5)
    ZERO32 = z3.BitVecVal(0, 32)
    
    rf_inv = (z3.Select(rf, ZERO5) == ZERO32)

    # Read helper
    def read_reg(regs, idx):
        return z3.If(idx == ZERO5, ZERO32, z3.Select(regs, idx))

    # Write helper
    def write_reg(regs, idx, val):
        new_regs = z3.Store(regs, idx, val)
        # Enforce x0 stays 0
        return z3.Store(new_regs, ZERO5, ZERO32)

    # Symbolic instruction fields
    rs1_idx = z3.BitVec('rs1_idx', 5)
    rs2_idx = z3.BitVec('rs2_idx', 5)
    rd_idx  = z3.BitVec('rd_idx', 5)
    imm12   = z3.BitVec('imm12', 12)

    # Symbolic operands
    rs1_val = read_reg(rf, rs1_idx)
    rs2_val = read_reg(rf, rs2_idx)
    shamt   = z3.ZeroExt(27, z3.Extract(4, 0, imm12))
    imm_ext = z3.SignExt(20, imm12)

    # Test ADD update
    rf_next_add = write_reg(rf, rd_idx, rs1_val + rs2_val)

    solver = z3.Solver()
    solver.add(rf_inv)

    # Prove x0 is still 0 after write to ANY rd_idx (including rd_idx == 0)
    solver.add(z3.Select(rf_next_add, ZERO5) != ZERO32)
    assert solver.check() == z3.unsat, "x0 invariant failed after ADD write"

    # Prove reading rd from rf_next_add gives correct sum when rd != 0
    solver.reset()
    solver.add(rf_inv)
    solver.add(rd_idx != ZERO5)
    solver.add(read_reg(rf_next_add, rd_idx) != rs1_val + rs2_val)
    assert solver.check() == z3.unsat, "ADD value check failed"

    print("Register file Z3 symbolic model verified successfully!")

if __name__ == "__main__":
    test_regfile_symbolic()
