#!/usr/bin/env python3
"""
equivalence_verifier.py — Formal Equivalence Verification via Z3 BitVector Symbolic Execution
with a strict 5-second timeout requirement.
"""

import time
import re
from typing import Union, List, Dict, Tuple, Any
import z3

from peephole_rulebook import OPCODE_MAP, REG_MAP

INT32_MIN = -(2**31)
INT32_MAX = (2**31) - 1

def _clamp_int32(val: int) -> int:
    return max(INT32_MIN, min(INT32_MAX, val))

def parse_assembly_inst_clean(line: str) -> List[int]:
    line = line.strip()
    if not line or line.startswith(".") or line.startswith("#") or line.endswith(":"):
        return [OPCODE_MAP["NOP"], 0, 0, 0, 0]
    if "#" in line:
        line = line.split("#")[0].strip()

    tokens = re.split(r'[\s,]+', line)
    op_str = tokens[0].upper()
    opcode_id = OPCODE_MAP.get(op_str, OPCODE_MAP.get("OTHER", 0))

    rd, rs1, rs2, imm = 0, 0, 0, 0
    if op_str == "LI" and len(tokens) >= 3:
        rd = REG_MAP.get(tokens[1], 0)
        try:
            imm = _clamp_int32(int(tokens[2], 0))
        except ValueError:
            imm = 0
    elif op_str in ("ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SLLI_I", "SRLI", "SRAI", "JALR") and len(tokens) >= 3:
        rd = REG_MAP.get(tokens[1], 0)
        if len(tokens) >= 4:
            rs1 = REG_MAP.get(tokens[2], 0)
            try:
                imm = _clamp_int32(int(tokens[3], 0))
            except ValueError:
                imm = 0
        else:
            try:
                imm = _clamp_int32(int(tokens[2], 0))
            except ValueError:
                imm = 0
    elif op_str in ("ADD", "SUB", "SLL", "SLT", "SLTU", "SRL", "SRA", "MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU", "AND", "OR", "XOR") and len(tokens) >= 4:
        rd = REG_MAP.get(tokens[1], 0)
        rs1 = REG_MAP.get(tokens[2], 0)
        rs2 = REG_MAP.get(tokens[3], 0)
    elif op_str == "MV" and len(tokens) >= 3:
        rd = REG_MAP.get(tokens[1], 0)
        rs1 = REG_MAP.get(tokens[2], 0)
    return [opcode_id, rs1, rs2, rd, imm]


def parse_asm_prog_clean(asm_text: str, max_len: int = 16) -> List[List[int]]:
    encoded = []
    for line in asm_text.splitlines():
        parsed = parse_assembly_inst_clean(line)
        if parsed[0] != OPCODE_MAP["NOP"] or line.strip() == "nop":
            encoded.append(parsed)
            if len(encoded) >= max_len:
                break
    while len(encoded) < max_len:
        encoded.append([OPCODE_MAP["NOP"], 0, 0, 0, 0])
    return encoded


def verify_equiv_status(
    original: Union[str, List[List[int]]],
    candidate: Union[str, List[List[int]]],
    timeout: float = 5.0
) -> str:
    """
    Formally verify equivalence between original and candidate assembly programs using Z3.
    Returns:
        "UNSAT"   - Formally proven equivalent for all inputs.
        "SAT"     - Proven non-equivalent (counterexample found).
        "TIMEOUT" - Execution exceeded timeout seconds.
        "UNKNOWN" - Solver returned unknown or encountered an exception.
    """
    if isinstance(original, str):
        prog_A = parse_asm_prog_clean(original, max_len=16)
    else:
        prog_A = original

    if isinstance(candidate, str):
        prog_B = parse_asm_prog_clean(candidate, max_len=16)
    else:
        prog_B = candidate

    solver = z3.Solver()
    solver.set("timeout", int(timeout * 1000))

    regs_A = [z3.BitVec(f"r{i}_init", 32) if i != 0 else z3.BitVecVal(0, 32) for i in range(32)]
    regs_B = list(regs_A)

    def execute_symbolic_program(prog: List[List[int]], initial_regs: List[Any]) -> List[Any]:
        regs = list(initial_regs)
        regs[0] = z3.BitVecVal(0, 32)

        for inst in prog:
            opcode, rs1, rs2, rd, imm = inst
            if opcode == OPCODE_MAP["NOP"]:
                continue
            elif opcode == OPCODE_MAP["LI"]:
                if rd != 0:
                    regs[rd] = z3.BitVecVal(imm & 0xFFFFFFFF, 32)
            elif opcode == OPCODE_MAP["ADD"]:
                if rd != 0:
                    regs[rd] = regs[rs1] + regs[rs2]
            elif opcode == OPCODE_MAP["ADDI"]:
                if rd != 0:
                    regs[rd] = regs[rs1] + z3.BitVecVal(imm & 0xFFFFFFFF, 32)
            elif opcode == OPCODE_MAP["SUB"]:
                if rd != 0:
                    regs[rd] = regs[rs1] - regs[rs2]
            elif opcode == OPCODE_MAP["MUL"]:
                if rd != 0:
                    regs[rd] = regs[rs1] * regs[rs2]
            elif opcode == OPCODE_MAP["DIV"]:
                if rd != 0:
                    div_val = z3.If(regs[rs2] == 0, z3.BitVecVal(0, 32), regs[rs1] / regs[rs2])
                    regs[rd] = div_val
            elif opcode in (OPCODE_MAP.get("DIVU", 36), 36):
                if rd != 0:
                    div_val = z3.If(regs[rs2] == 0, z3.BitVecVal(0, 32), z3.UDiv(regs[rs1], regs[rs2]))
                    regs[rd] = div_val
            elif opcode in (OPCODE_MAP["SLLI"], OPCODE_MAP.get("SLLI_I", OPCODE_MAP["SLLI"])):
                if rd != 0:
                    regs[rd] = regs[rs1] << (imm & 0x1F)
            elif opcode == OPCODE_MAP["SRLI"]:
                if rd != 0:
                    regs[rd] = z3.LShR(regs[rs1], imm & 0x1F)
            elif opcode == OPCODE_MAP["SRAI"]:
                if rd != 0:
                    regs[rd] = regs[rs1] >> (imm & 0x1F)
            elif opcode == OPCODE_MAP["AND"]:
                if rd != 0:
                    regs[rd] = regs[rs1] & regs[rs2]
            elif opcode == OPCODE_MAP["ANDI"]:
                if rd != 0:
                    regs[rd] = regs[rs1] & z3.BitVecVal(imm & 0xFFFFFFFF, 32)
            elif opcode == OPCODE_MAP["OR"]:
                if rd != 0:
                    regs[rd] = regs[rs1] | regs[rs2]
            elif opcode == OPCODE_MAP["ORI"]:
                if rd != 0:
                    regs[rd] = regs[rs1] | z3.BitVecVal(imm & 0xFFFFFFFF, 32)
            elif opcode == OPCODE_MAP["XOR"]:
                if rd != 0:
                    regs[rd] = regs[rs1] ^ regs[rs2]
            elif opcode == OPCODE_MAP["XORI"]:
                if rd != 0:
                    regs[rd] = regs[rs1] ^ z3.BitVecVal(imm & 0xFFFFFFFF, 32)
            elif opcode == OPCODE_MAP["MV"]:
                if rd != 0:
                    regs[rd] = regs[rs1]

            regs[0] = z3.BitVecVal(0, 32)

        return regs

    final_A = execute_symbolic_program(prog_A, regs_A)
    final_B = execute_symbolic_program(prog_B, regs_B)

    discrepancy_conditions = [final_A[r] != final_B[r] for r in range(1, 32)]
    solver.add(z3.Or(discrepancy_conditions))

    start_time = time.time()
    try:
        check_res = solver.check()
        elapsed = time.time() - start_time
        if elapsed > timeout or check_res == z3.unknown:
            return "TIMEOUT" if elapsed > timeout else "UNKNOWN"

        if check_res == z3.unsat:
            return "UNSAT"
        elif check_res == z3.sat:
            return "SAT"
        else:
            return "UNKNOWN"
    except Exception as e:
        print(f" [verify_equiv] Verification Exception: {e}")
        return "UNKNOWN"


def verify_equiv(
    original: Union[str, List[List[int]]],
    candidate: Union[str, List[List[int]]],
    timeout: float = 5.0
) -> bool:
    return verify_equiv_status(original, candidate, timeout=timeout) == "UNSAT"


if __name__ == "__main__":
    print(" Testing Equivalence Verifier without Circular Dependencies...")
    p_orig = "li t0, 4\nmul t1, a0, t0"
    p_cand = "slli t1, a0, 2"
    assert verify_equiv(p_orig, p_cand, timeout=5.0) == True
    print(" ✓ Equivalence verifier clean test passed!")
