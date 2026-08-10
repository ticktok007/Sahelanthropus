import z3
from typing import List, Tuple, Union

# Sorts & Constants
BV5  = z3.BitVecSort(5)
BV32 = z3.BitVecSort(32)
BV8  = z3.BitVecSort(8)

ZERO5  = z3.BitVecVal(0, 5)
ZERO32 = z3.BitVecVal(0, 32)

REG_MAP = {f"x{i}": i for i in range(32)}
REG_MAP.update({
    "zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15,
    "a6": 16, "a7": 17, "s2": 18, "s3": 19, "s4": 20, "s5": 21,
    "s6": 22, "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31
})

def parse_reg(reg: Union[str, int]) -> z3.BitVecNumRef:
    if isinstance(reg, int):
        return z3.BitVecVal(reg & 0x1F, 5)
    reg_str = reg.lower().strip()
    idx = REG_MAP.get(reg_str, 0)
    return z3.BitVecVal(idx, 5)

def read_reg(regs, reg_idx):
    return z3.If(reg_idx == ZERO5, ZERO32, z3.Select(regs, reg_idx))

def write_reg(regs, reg_idx, val):
    updated = z3.Store(regs, reg_idx, val)
    return z3.Store(updated, ZERO5, ZERO32)

def execute_program(instrs: List[Tuple], rf_init, mem_init):
    rf = rf_init
    mem = mem_init

    for instr in instrs:
        op = instr[0].upper()
        
        if op in ("ADD", "SUB", "AND", "OR", "XOR"):
            rd, rs1, rs2 = parse_reg(instr[1]), parse_reg(instr[2]), parse_reg(instr[3])
            v1, v2 = read_reg(rf, rs1), read_reg(rf, rs2)
            if op == "ADD": res = v1 + v2
            elif op == "SUB": res = v1 - v2
            elif op == "AND": res = v1 & v2
            elif op == "OR":  res = v1 | v2
            elif op == "XOR": res = v1 ^ v2
            rf = write_reg(rf, rd, res)

        elif op in ("ADDI", "ANDI", "ORI", "XORI", "SLLI", "SRLI", "SRAI"):
            rd, rs1, imm = parse_reg(instr[1]), parse_reg(instr[2]), instr[3]
            v1 = read_reg(rf, rs1)
            if op in ("SLLI", "SRLI", "SRAI"):
                shamt = z3.ZeroExt(27, z3.BitVecVal(imm & 0x1F, 5))
                if op == "SLLI": res = v1 << shamt
                elif op == "SRLI": res = z3.LShR(v1, shamt)
                elif op == "SRAI": res = v1 >> shamt
            else:
                imm_bv = z3.SignExt(20, z3.BitVecVal(imm & 0xFFF, 12))
                if op == "ADDI": res = v1 + imm_bv
                elif op == "ANDI": res = v1 & imm_bv
                elif op == "ORI":  res = v1 | imm_bv
                elif op == "XORI": res = v1 ^ imm_bv
            rf = write_reg(rf, rd, res)

        elif op == "LW":
            rd, imm, rs1 = parse_reg(instr[1]), instr[2], parse_reg(instr[3])
            addr = read_reg(rf, rs1) + z3.SignExt(20, z3.BitVecVal(imm & 0xFFF, 12))
            val = z3.Concat(
                z3.Select(mem, addr + 3),
                z3.Select(mem, addr + 2),
                z3.Select(mem, addr + 1),
                z3.Select(mem, addr)
            )
            rf = write_reg(rf, rd, val)

        elif op == "SW":
            rs2, imm, rs1 = parse_reg(instr[1]), instr[2], parse_reg(instr[3])
            addr = read_reg(rf, rs1) + z3.SignExt(20, z3.BitVecVal(imm & 0xFFF, 12))
            val = read_reg(rf, rs2)
            mem = z3.Store(
                z3.Store(
                    z3.Store(
                        z3.Store(mem, addr, z3.Extract(7, 0, val)),
                        addr + 1, z3.Extract(15, 8, val)
                    ),
                    addr + 2, z3.Extract(23, 16, val)
                ),
                addr + 3, z3.Extract(31, 24, val)
            )

    return rf, mem

def verify_equiv(prog_A_instrs: List[Tuple], prog_B_instrs: List[Tuple]) -> bool:
    """
    Symbolically executes prog_A and prog_B from the same initial state.
    Returns True if prog_A_instrs ≡ prog_B_instrs for all inputs, False otherwise.
    """
    rf_init = z3.Array('rf_init', BV5, BV32)
    mem_init = z3.Array('mem_init', BV32, BV8)

    solver = z3.Solver()
    # Invariant: x0 is always 0 in initial register file
    solver.add(z3.Select(rf_init, ZERO5) == ZERO32)

    rf_A, mem_A = execute_program(prog_A_instrs, rf_init, mem_init)
    rf_B, mem_B = execute_program(prog_B_instrs, rf_init, mem_init)

    # Check if final state differs for any register or memory location
    diff_reg = z3.Exists([z3.BitVec('i', 5)], z3.Select(rf_A, z3.BitVec('i', 5)) != z3.Select(rf_B, z3.BitVec('i', 5)))
    diff_mem = z3.Exists([z3.BitVec('a', 32)], z3.Select(mem_A, z3.BitVec('a', 32)) != z3.Select(mem_B, z3.BitVec('a', 32)))

    solver.add(z3.Or(diff_reg, diff_mem))
    
    # UNSAT means no state difference exists -> Equivalent!
    return solver.check() == z3.unsat

# Tests
if __name__ == "__main__":
    # Test 1: SLLI x1, x2, 1  ≡  ADD x1, x2, x2
    p1 = [("SLLI", "x1", "x2", 1)]
    p2 = [("ADD", "x1", "x2", "x2")]
    res1 = verify_equiv(p1, p2)
    print(f"Test 1 (SLLI x1,x2,1 vs ADD x1,x2,x2): {res1}")
    assert res1 == True

    # Test 2: SLLI x1, x2, 2  ≢  ADDI x1, x2, 4
    p3 = [("SLLI", "x1", "x2", 2)]
    p4 = [("ADDI", "x1", "x2", 4)]
    res2 = verify_equiv(p3, p4)
    print(f"Test 2 (SLLI x1,x2,2 vs ADDI x1,x2,4): {res2}")
    assert res2 == False

    # Test 3: Algebraic simplification (x1 = x2 + x2 + x2  vs  x1 = 3 * x2 via ADD + ADD)
    p5 = [("ADD", "t0", "x2", "x2"), ("ADD", "x1", "t0", "x2")]
    p6 = [("SLLI", "t0", "x2", 1), ("ADD", "x1", "t0", "x2")]
    res3 = verify_equiv(p5, p6)
    print(f"Test 3 (Chained ADDs vs SLLI+ADD): {res3}")
    assert res3 == True

    print("\nAll verify_equiv tests passed successfully!")
