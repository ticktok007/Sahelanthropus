import z3
from typing import List, Tuple, Union

# Z3 Sort Definitions
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

def _parse_reg(reg: Union[str, int]) -> z3.BitVecNumRef:
    if isinstance(reg, int):
        return z3.BitVecVal(reg & 0x1F, 5)
    return z3.BitVecVal(REG_MAP.get(str(reg).lower().strip(), 0), 5)

def _read_reg(regs, reg_idx):
    return z3.If(reg_idx == ZERO5, ZERO32, z3.Select(regs, reg_idx))

def _write_reg(regs, reg_idx, val):
    updated = z3.Store(regs, reg_idx, val)
    return z3.Store(updated, ZERO5, ZERO32)

def _execute_program(instrs: List[Tuple], rf_init, mem_init):
    rf = rf_init
    mem = mem_init

    for instr in instrs:
        op = instr[0].upper()

        if op in ("ADD", "SUB", "AND", "OR", "XOR"):
            rd, rs1, rs2 = _parse_reg(instr[1]), _parse_reg(instr[2]), _parse_reg(instr[3])
            v1, v2 = _read_reg(rf, rs1), _read_reg(rf, rs2)
            if op == "ADD":   res = v1 + v2
            elif op == "SUB": res = v1 - v2
            elif op == "AND": res = v1 & v2
            elif op == "OR":  res = v1 | v2
            elif op == "XOR": res = v1 ^ v2
            rf = _write_reg(rf, rd, res)

        elif op in ("ADDI", "ANDI", "ORI", "XORI", "SLLI", "SRLI", "SRAI"):
            rd, rs1, imm = _parse_reg(instr[1]), _parse_reg(instr[2]), instr[3]
            v1 = _read_reg(rf, rs1)
            if op in ("SLLI", "SRLI", "SRAI"):
                shamt = z3.ZeroExt(27, z3.BitVecVal(imm & 0x1F, 5))
                if op == "SLLI":   res = v1 << shamt
                elif op == "SRLI": res = z3.LShR(v1, shamt)
                elif op == "SRAI": res = v1 >> shamt
            else:
                imm_bv = z3.SignExt(20, z3.BitVecVal(imm & 0xFFF, 12))
                if op == "ADDI":   res = v1 + imm_bv
                elif op == "ANDI": res = v1 & imm_bv
                elif op == "ORI":  res = v1 | imm_bv
                elif op == "XORI": res = v1 ^ imm_bv
            rf = _write_reg(rf, rd, res)

        elif op == "LW":
            rd, imm, rs1 = _parse_reg(instr[1]), instr[2], _parse_reg(instr[3])
            addr = _read_reg(rf, rs1) + z3.SignExt(20, z3.BitVecVal(imm & 0xFFF, 12))
            val = z3.Concat(
                z3.Select(mem, addr + 3),
                z3.Select(mem, addr + 2),
                z3.Select(mem, addr + 1),
                z3.Select(mem, addr)
            )
            rf = _write_reg(rf, rd, val)

        elif op == "SW":
            rs2, imm, rs1 = _parse_reg(instr[1]), instr[2], _parse_reg(instr[3])
            addr = _read_reg(rf, rs1) + z3.SignExt(20, z3.BitVecVal(imm & 0xFFF, 12))
            val = _read_reg(rf, rs2)
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
    rf_init  = z3.Array('rf_init',  BV5,  BV32)
    mem_init = z3.Array('mem_init', BV32, BV8)

    solver = z3.Solver()
    solver.add(z3.Select(rf_init, ZERO5) == ZERO32)

    rf_A, mem_A = _execute_program(prog_A_instrs, rf_init, mem_init)
    rf_B, mem_B = _execute_program(prog_B_instrs, rf_init, mem_init)

    r_idx  = z3.BitVec('r_idx', 5)
    m_addr = z3.BitVec('m_addr', 32)

    diff_reg = z3.Exists([r_idx],  z3.Select(rf_A, r_idx)   != z3.Select(rf_B, r_idx))
    diff_mem = z3.Exists([m_addr], z3.Select(mem_A, m_addr) != z3.Select(mem_B, m_addr))

    solver.add(z3.Or(diff_reg, diff_mem))
    return solver.check() == z3.unsat

# ---------------------------------------------------------------------------
# Test Suite: 20 Equivalent Pairs & 5 Inequivalent Pairs
# ---------------------------------------------------------------------------

EQUIVALENT_PAIRS = [
    # 1. SLLI 1 vs ADD self
    ([("SLLI", "x1", "x2", 1)], [("ADD", "x1", "x2", "x2")]),
    # 2. ADD commutativity
    ([("ADD", "x1", "x2", "x3")], [("ADD", "x1", "x3", "x2")]),
    # 3. AND commutativity
    ([("AND", "x1", "x2", "x3")], [("AND", "x1", "x3", "x2")]),
    # 4. OR commutativity
    ([("OR", "x1", "x2", "x3")], [("OR", "x1", "x3", "x2")]),
    # 5. XOR commutativity
    ([("XOR", "x1", "x2", "x3")], [("XOR", "x1", "x3", "x2")]),
    # 6. XOR self is zero
    ([("XOR", "x1", "x2", "x2")], [("ADDI", "x1", "zero", 0)]),
    # 7. SUB self is zero
    ([("SUB", "x1", "x2", "x2")], [("ADDI", "x1", "zero", 0)]),
    # 8. ADD zero
    ([("ADD", "x1", "x2", "zero")], [("ADDI", "x1", "x2", 0)]),
    # 9. OR zero
    ([("OR", "x1", "x2", "zero")], [("ADDI", "x1", "x2", 0)]),
    # 10. Chained ADD (3 * x2)
    ([("ADD", "t0", "x2", "x2"), ("ADD", "x1", "t0", "x2")], [("SLLI", "t0", "x2", 1), ("ADD", "x1", "t0", "x2")]),
    # 11. Chained SLLI 1+1 vs SLLI 2
    ([("SLLI", "t0", "x2", 1), ("SLLI", "x1", "t0", 1)], [("SLLI", "t0", "x2", 1), ("SLLI", "x1", "x2", 2)]),
    # 12. Chained ADDI 5+10 vs ADDI 15
    ([("ADDI", "t0", "x2", 5), ("ADDI", "x1", "t0", 10)], [("ADDI", "t0", "x2", 5), ("ADDI", "x1", "x2", 15)]),
    # 13. Dead write overwrite
    ([("ADDI", "x1", "x2", 10), ("ADDI", "x1", "x2", 20)], [("ADDI", "x1", "x2", 20)]),
    # 14. Write to x0 (zero register) is discarded
    ([("ADD", "x0", "x2", "x3")], [("SUB", "x0", "x2", "x3")]),
    # 15. ANDI -1 identity
    ([("ANDI", "x1", "x2", -1)], [("ADDI", "x1", "x2", 0)]),
    # 16. Store word then Load word
    ([("SW", "x2", 0, "sp"), ("LW", "x1", 0, "sp")], [("SW", "x2", 0, "sp"), ("ADDI", "x1", "x2", 0)]),
    # 17. SRLI then SLLI (bitmasking lower 2 bits)
    ([("SRLI", "t0", "x2", 2), ("SLLI", "x1", "t0", 2)], [("SRLI", "t0", "x2", 2), ("ANDI", "x1", "x2", -4)]),
    # 18. SUB via ADDI negative constant
    ([("ADDI", "t0", "zero", 5), ("SUB", "x1", "x2", "t0")], [("ADDI", "t0", "zero", 5), ("ADDI", "x1", "x2", -5)]),
    # 19. Identity subtraction across different source registers (x1 = a0 - a0 vs x1 = a1 - a1)
    ([("SUB", "x1", "a0", "a0")], [("SUB", "x1", "a1", "a1")]),
    # 20. ANDI 15 vs ADDI + AND
    ([("ANDI", "x1", "x2", 15), ("ADDI", "t0", "zero", 15)], [("ADDI", "t0", "zero", 15), ("AND", "x1", "x2", "t0")]),
]

INEQUIVALENT_PAIRS = [
    # 1. SLLI 2 vs ADDI 4
    ([("SLLI", "x1", "x2", 2)], [("ADDI", "x1", "x2", 4)]),
    # 2. ADD vs SUB
    ([("ADD", "x1", "x2", "x3")], [("SUB", "x1", "x2", "x3")]),
    # 3. SRLI vs SRAI (differs on negative inputs)
    ([("SRLI", "x1", "x2", 2)], [("SRAI", "x1", "x2", 2)]),
    # 4. Different immediate constant (ADDI 10 vs ADDI 11)
    ([("ADDI", "x1", "x2", 10)], [("ADDI", "x1", "x2", 11)]),
    # 5. Different target register (x1 vs x4)
    ([("ADD", "x1", "x2", "x3")], [("ADD", "x4", "x2", "x3")]),
]

def run_suite():
    passed_eq = 0
    passed_ineq = 0

    print("===============================================================")
    print(" Running Equivalence Verification Test Suite (25 Test Pairs)")
    print("===============================================================\n")

    print("--- 20 Known-Equivalent Pairs ---")
    for idx, (pA, pB) in enumerate(EQUIVALENT_PAIRS, 1):
        res = verify_equiv(pA, pB)
        status = "PASS" if res is True else "FAIL"
        if res is True:
            passed_eq += 1
        print(f"  Pair {idx:02d}: [{status}] Expected True, Got {res}")

    print("\n--- 5 Known-Inequivalent Pairs ---")
    for idx, (pA, pB) in enumerate(INEQUIVALENT_PAIRS, 1):
        res = verify_equiv(pA, pB)
        status = "PASS" if res is False else "FAIL"
        if res is False:
            passed_ineq += 1
        print(f"  Pair {idx:02d}: [{status}] Expected False, Got {res}")

    print("\n===============================================================")
    print(f" Summary: {passed_eq}/20 Equivalent PASS, {passed_ineq}/5 Inequivalent PASS")
    print("===============================================================")

    assert passed_eq == 20, f"Failed equivalent tests: {passed_eq}/20"
    assert passed_ineq == 5, f"Failed inequivalent tests: {passed_ineq}/5"

if __name__ == "__main__":
    run_suite()
