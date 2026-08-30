#!/usr/bin/env python3
"""
asm_graph_builder.py — Construct PyTorch Geometric HeteroData graphs with 177-dim node features
and explicit Control Flow + Def-Use Data Flow + Register Read/Write edges (including reverse relations).
"""

from typing import List, Tuple, Dict, Set
import re
import torch
import numpy as np

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from superopt_env import parse_assembly_program, INT32_MIN, INT32_MAX, _clamp_int32

# 47 RISC-V Opcodes Vocabulary Map
OPCODE_47_MAP: Dict[str, int] = {
    "NOP": 0, "LI": 1, "ADD": 2, "ADDI": 3, "SUB": 4, "MUL": 5, "DIV": 6,
    "REM": 7, "DIVU": 8, "REMU": 9, "SLLI": 10, "SRLI": 11, "SRAI": 12,
    "SLL": 13, "SRL": 14, "SRA": 15, "AND": 16, "ANDI": 17, "OR": 18,
    "ORI": 19, "XOR": 20, "XORI": 21, "MV": 22, "LB": 23, "LH": 24,
    "LW": 25, "LD": 26, "LBU": 27, "LHU": 28, "LWU": 29, "SB": 30,
    "SH": 31, "SW": 32, "SD": 33, "BEQ": 34, "BNE": 35, "BLT": 36,
    "BGE": 37, "BLTU": 38, "BGEU": 39, "J": 40, "JAL": 41, "JALR": 42,
    "SLTI": 43, "SLTIU": 44, "SLLI_I": 10, "MULH": 5, "MULHSU": 5, "MULHU": 5, "ECALL": 45, "LUI": 46, "OTHER": 47
}

BRANCH_OPCODES = {"BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU", "J", "JAL", "JALR"}
MEM_OPCODES = {"LB", "LH", "LW", "LD", "LBU", "LHU", "LWU", "SB", "SH", "SW", "SD"}


def encode_instruction_177dim(inst: List[int], op_str: str = "OTHER") -> torch.Tensor:
    opcode_id, rs1, rs2, rd, imm = int(inst[0]), int(inst[1]), int(inst[2]), int(inst[3]), int(inst[4])

    # 1. Opcode one-hot (47-dim)
    op_onehot = np.zeros(47, dtype=np.float32)
    op_idx = min(46, max(0, opcode_id))
    op_onehot[op_idx] = 1.0

    rd_onehot = np.zeros(32, dtype=np.float32)
    if 0 <= rd < 32:
        rd_onehot[rd] = 1.0

    rs1_onehot = np.zeros(32, dtype=np.float32)
    if 0 <= rs1 < 32:
        rs1_onehot[rs1] = 1.0

    rs2_onehot = np.zeros(32, dtype=np.float32)
    if 0 <= rs2 < 32:
        rs2_onehot[rs2] = 1.0

    imm_clamped = _clamp_int32(imm) & 0xFFFFFFFF
    imm_bits = np.array([(imm_clamped >> k) & 1 for k in range(31, -1, -1)], dtype=np.float32)

    op_upper = op_str.upper()
    is_branch = np.array([1.0 if op_upper in BRANCH_OPCODES else 0.0], dtype=np.float32)
    is_mem = np.array([1.0 if op_upper in MEM_OPCODES else 0.0], dtype=np.float32)

    feature_vec = np.concatenate([
        op_onehot, rd_onehot, rs1_onehot, rs2_onehot, imm_bits, is_branch, is_mem
    ], axis=0)

    return torch.tensor(feature_vec, dtype=torch.float32)


def parse_assembly_with_labels(asm_text: str, max_len: int = 16) -> Tuple[List[List[int]], List[str], Dict[str, int]]:
    lines = []
    label_map: Dict[str, int] = {}
    
    inst_count = 0
    for line in asm_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(".") or stripped.startswith("#"):
            continue
        if stripped.endswith(":"):
            label_name = stripped[:-1].strip()
            label_map[label_name] = inst_count
            continue
        if ":" in stripped:
            parts = stripped.split(":")
            label_name = parts[0].strip()
            label_map[label_name] = inst_count
            stripped = parts[1].strip()

        lines.append(stripped)
        inst_count += 1
        if inst_count >= max_len:
            break

    prog = parse_assembly_program(asm_text, max_len=max_len)
    return prog, lines, label_map


def asm_to_graph(asm_text: str, max_len: int = 16) -> "HeteroData":
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for HeteroData graph construction.")

    prog, raw_lines, label_map = parse_assembly_with_labels(asm_text, max_len=max_len)
    num_insts = len(prog)

    inst_features = []
    for i, inst in enumerate(prog):
        op_str = raw_lines[i].split()[0] if i < len(raw_lines) else "OTHER"
        vec = encode_instruction_177dim(inst, op_str=op_str)
        inst_features.append(vec)

    inst_x = torch.stack(inst_features, dim=0)
    reg_x = torch.arange(32, dtype=torch.float32).unsqueeze(-1)

    control_flow_edges: Set[Tuple[int, int]] = set()
    data_flow_edges: Set[Tuple[int, int]] = set()
    reads_edges: Set[Tuple[int, int]] = set()
    read_by_edges: Set[Tuple[int, int]] = set()
    writes_edges: Set[Tuple[int, int]] = set()
    written_by_edges: Set[Tuple[int, int]] = set()

    last_def: Dict[int, int] = {}

    for i, inst in enumerate(prog):
        opcode_id, rs1, rs2, rd, imm = inst
        op_str = raw_lines[i].split()[0].upper() if i < len(raw_lines) else ""

        if i < num_insts - 1:
            control_flow_edges.add((i, i + 1))

        if op_str in BRANCH_OPCODES:
            tokens = re.split(r'[\s,]+', raw_lines[i])
            for token in tokens[1:]:
                if token in label_map:
                    target_slot = label_map[token]
                    if 0 <= target_slot < num_insts:
                        control_flow_edges.add((i, target_slot))

        if rs1 > 0:
            reads_edges.add((i, rs1))
            read_by_edges.add((rs1, i))  # reg -> inst
            if rs1 in last_def:
                data_flow_edges.add((last_def[rs1], i))

        if rs2 > 0:
            reads_edges.add((i, rs2))
            read_by_edges.add((rs2, i))  # reg -> inst
            if rs2 in last_def:
                data_flow_edges.add((last_def[rs2], i))

        if rd > 0:
            writes_edges.add((i, rd))
            written_by_edges.add((rd, i))  # reg -> inst
            last_def[rd] = i

    def make_edge_tensor(edge_set: Set[Tuple[int, int]]) -> torch.Tensor:
        if not edge_set:
            return torch.empty((2, 0), dtype=torch.long)
        sorted_list = sorted(list(edge_set))
        return torch.tensor(sorted_list, dtype=torch.long).t().contiguous()

    data = HeteroData()

    data['inst'].x = inst_x
    data['reg'].x = reg_x

    data['inst', 'control_flow', 'inst'].edge_index = make_edge_tensor(control_flow_edges)
    data['inst', 'data_flow', 'inst'].edge_index = make_edge_tensor(data_flow_edges)
    data['inst', 'reads', 'reg'].edge_index = make_edge_tensor(reads_edges)
    data['reg', 'read_by', 'inst'].edge_index = make_edge_tensor(read_by_edges)
    data['inst', 'writes', 'reg'].edge_index = make_edge_tensor(writes_edges)
    data['reg', 'written_by', 'inst'].edge_index = make_edge_tensor(written_by_edges)

    return data
