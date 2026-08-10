import z3
from typing import List, Tuple, Union, Dict

BV5  = z3.BitVecSort(5)   # 5-bit Register Index (0..31)
BV32 = z3.BitVecSort(32)  # 32-bit BitVector
BV8  = z3.BitVecSort(8)   # 8-bit Byte

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
    """
    Passes the EXACT SAME initial symbolic register file state `rf_init`
    and initial memory state `mem_init` to both programs.
    """
    # 1. Create a single shared initial register file state Array
    rf_init  = z3.Array('rf_init', BV5, BV32)
    mem_init = z3.Array('mem_init', BV32, BV8)

    solver = z3.Solver()
    
    # 2. Hardwired x0 == 0 constraint on initial register state
    solver.add(z3.Select(rf_init, ZERO5) == ZERO32)

    # 3. Symbolically execute BOTH programs from identical rf_init
    rf_A, mem_A = _execute_program(prog_A_instrs, rf_init, mem_init)
    rf_B, mem_B = _execute_program(prog_B_instrs, rf_init, mem_init)

    # 4. Formulate equality check over ALL registers and memory locations
    r_idx  = z3.BitVec('r_idx', 5)
    m_addr = z3.BitVec('m_addr', 32)
    
    diff_reg = z3.Exists([r_idx], z3.Select(rf_A, r_idx) != z3.Select(rf_B, r_idx))
    diff_mem = z3.Exists([m_addr], z3.Select(mem_A, m_addr) != z3.Select(mem_B, m_addr))

    solver.add(z3.Or(diff_reg, diff_mem))

    # UNSAT -> No initial register state exists where outputs differ -> Equivalent!
    return solver.check() == z3.unsat

if __name__ == "__main__":
    # Test case: Register persistence
    # Program A modifies x1, leaves x2..x31 untouched
    # Program B modifies x1 in a different way, but result is identical
    prog1 = [("ADD", "x1", "x2", "x3")]
    prog2 = [("ADD", "x1", "x3", "x2")]  # Commutative addition
    assert verify_equiv(prog1, prog2) == True
    print("Shared initial state test passed!")
