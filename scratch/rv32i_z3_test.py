import z3

def rv32i_z3_formulas():
    BV32 = z3.BitVecSort(32)
    BV8 = z3.BitVecSort(8)
    
    rs1 = z3.BitVec('rs1', 32)
    rs2 = z3.BitVec('rs2', 32)
    imm12 = z3.BitVec('imm12', 12)
    
    shamt = z3.ZeroExt(27, z3.Extract(4, 0, imm12))
    imm12_sext = z3.SignExt(20, imm12)
    
    mem = z3.Array('mem', BV32, BV8)
    
    formulas = {
        "ADD":  rs1 + rs2,
        "SUB":  rs1 - rs2,
        "AND":  rs1 & rs2,
        "OR":   rs1 | rs2,
        "XOR":  rs1 ^ rs2,
        "SLLI": rs1 << shamt,
        "SRLI": z3.LShR(rs1, shamt),
        "SRAI": rs1 >> shamt,
        "LW":   z3.Concat(
                    z3.Select(mem, rs1 + imm12_sext + 3),
                    z3.Select(mem, rs1 + imm12_sext + 2),
                    z3.Select(mem, rs1 + imm12_sext + 1),
                    z3.Select(mem, rs1 + imm12_sext)
                ),
        "SW":   z3.Store(
                    z3.Store(
                        z3.Store(
                            z3.Store(mem, rs1 + imm12_sext, z3.Extract(7, 0, rs2)),
                            rs1 + imm12_sext + 1, z3.Extract(15, 8, rs2)
                        ),
                        rs1 + imm12_sext + 2, z3.Extract(23, 16, rs2)
                    ),
                    rs1 + imm12_sext + 3, z3.Extract(31, 24, rs2)
                )
    }

    solver = z3.Solver()
    
    # Test ADD identity
    solver.add(rs1 == 10, rs2 == 20)
    solver.add(formulas["ADD"] == 30)
    assert solver.check() == z3.sat, "ADD formula failed"
    
    # Test SRAI sign extension
    solver.reset()
    solver.add(rs1 == z3.BitVecVal(-16, 32), imm12 == 2)
    solver.add(formulas["SRAI"] == z3.BitVecVal(-4, 32))
    assert solver.check() == z3.sat, "SRAI formula failed"
    
    # Test SW followed by LW load-after-store consistency
    solver.reset()
    addr = rs1 + imm12_sext
    mem_after_sw = formulas["SW"]
    loaded_val = z3.Concat(
        z3.Select(mem_after_sw, addr + 3),
        z3.Select(mem_after_sw, addr + 2),
        z3.Select(mem_after_sw, addr + 1),
        z3.Select(mem_after_sw, addr)
    )
    solver.add(loaded_val != rs2)
    assert solver.check() == z3.unsat, "Store-Load consistency failed"

    print("All 10 RV32I Z3 formulas verified successfully!")

if __name__ == "__main__":
    rv32i_z3_formulas()
