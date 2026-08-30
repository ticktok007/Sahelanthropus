#!/usr/bin/env python3
"""
test_rv32i_encoding.py — Verify encoding of all remaining RV32I instructions:
ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI_I, BEQ, BNE, BLT, BGE, BLTU, BGEU, JAL, JALR
"""

from superopt_env import parse_assembly_instruction, parse_assembly_program, OPCODE_MAP
from asm_graph_builder import encode_instruction_177dim, asm_to_graph

TEST_INSTRUCTIONS = [
    ("addi t0, t1, 42", "ADDI"),
    ("slti t0, t1, -5", "SLTI"),
    ("sltiu t0, t1, 10", "SLTIU"),
    ("xori t0, t1, 15", "XORI"),
    ("ori t0, t1, 255", "ORI"),
    ("andi t0, t1, 127", "ANDI"),
    ("slli t0, t1, 3", "SLLI"),
    ("slli_i t0, t1, 4", "SLLI_I"),
    ("beq t0, t1, 8", "BEQ"),
    ("bne t0, t1, 12", "BNE"),
    ("blt t0, t1, 16", "BLT"),
    ("bge t0, t1, 20", "BGE"),
    ("bltu t0, t1, 24", "BLTU"),
    ("bgeu t0, t1, 28", "BGEU"),
    ("jal ra, 32", "JAL"),
    ("jalr ra, t0, 0", "JALR"),
]


def test_rv32i_encoding():
    print("=" * 70)
    print(" TESTING RV32I INSTRUCTION ENCODING")
    print("=" * 70)

    for asm_line, op_name in TEST_INSTRUCTIONS:
        parsed = parse_assembly_instruction(asm_line)
        opcode_id, rs1, rs2, rd, imm = parsed
        expected_op = op_name if op_name in OPCODE_MAP else ("SLLI" if op_name == "SLLI_I" else op_name)
        assert opcode_id == OPCODE_MAP.get(expected_op, OPCODE_MAP.get(op_name, 0)), f"Failed mapping for {asm_line}"
        print(f" ✓ [Parsed] '{asm_line:<20}' -> Opcode: {opcode_id:>2d} ({expected_op}), rs1: {rs1:>2d}, rs2: {rs2:>2d}, rd: {rd:>2d}, imm: {imm}")

    # Test full assembly program parsing & PyG graph construction
    sample_asm = "\n".join([line for line, _ in TEST_INSTRUCTIONS])
    program = parse_assembly_program(sample_asm, max_len=16)
    assert len(program) == 16, f"Expected 16 instructions, got {len(program)}"

    graph = asm_to_graph(sample_asm, max_len=16)
    assert graph['inst'].x.shape == (16, 177), f"Expected graph inst features (16, 177), got {graph['inst'].x.shape}"

    print(f"\n ✓ [PyG Graph Construction] HeteroData built successfully with shape {graph['inst'].x.shape}")
    print("=" * 70)
    print(" ALL RV32I INSTRUCTION ENCODING CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_rv32i_encoding()
