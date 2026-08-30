#!/usr/bin/env python3
"""
test_rv32m_encoding.py — Verify encoding of all 8 RV32M Extension instructions:
MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
"""

from superopt_env import parse_assembly_instruction, parse_assembly_program, OPCODE_MAP
from asm_graph_builder import asm_to_graph

RV32M_INSTRUCTIONS = [
    ("mul t0, t1, t2", "MUL"),
    ("mulh t0, t1, t2", "MULH"),
    ("mulhsu t0, t1, t2", "MULHSU"),
    ("mulhu t0, t1, t2", "MULHU"),
    ("div t0, t1, t2", "DIV"),
    ("divu t0, t1, t2", "DIVU"),
    ("rem t0, t1, t2", "REM"),
    ("remu t0, t1, t2", "REMU"),
]


def test_rv32m_encoding():
    print("=" * 70)
    print(" TESTING RV32M EXTENSION INSTRUCTION ENCODING")
    print("=" * 70)

    for asm_line, op_name in RV32M_INSTRUCTIONS:
        parsed = parse_assembly_instruction(asm_line)
        opcode_id, rs1, rs2, rd, imm = parsed
        assert opcode_id == OPCODE_MAP[op_name], f"Failed opcode mapping for {asm_line}"
        assert rs1 == 6 and rs2 == 7 and rd == 5, f"Operand parsing mismatch for {asm_line}"
        print(f" ✓ [Parsed] '{asm_line:<20}' -> Opcode: {opcode_id:>2d} ({op_name}), rs1: {rs1:>2d} (t1), rs2: {rs2:>2d} (t2), rd: {rd:>2d} (t0)")

    sample_asm = "\n".join([line for line, _ in RV32M_INSTRUCTIONS])
    program = parse_assembly_program(sample_asm, max_len=16)
    graph = asm_to_graph(sample_asm, max_len=16)

    assert graph['inst'].x.shape[0] == 16
    print(f"\n ✓ [PyG Graph Construction] HeteroData graph built successfully with shape {graph['inst'].x.shape}")
    print("=" * 70)
    print(" ALL 8 RV32M EXTENSION INSTRUCTIONS VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_rv32m_encoding()
