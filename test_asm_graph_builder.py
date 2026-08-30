#!/usr/bin/env python3
"""
test_asm_graph_builder.py — Unit Tests for Control Flow + Def->Use Data Flow HeteroData graphs
"""

import unittest
import torch
from asm_graph_builder import asm_to_graph, encode_instruction_177dim, HAS_PYG


class TestAsmGraphBuilderEdges(unittest.TestCase):

    def setUp(self):
        if not HAS_PYG:
            self.skipTest("PyTorch Geometric is not installed.")

        self.sample_branch_asm = """
        .section .text
        .globl _start
        _start:
            li t0, 0
            li t1, 10
        loop_start:
            addi t0, t0, 1
            blt t0, t1, loop_start
            mv a0, t0
            li a7, 93
            ecall
        """

    def test_control_flow_edges(self):
        """Test Control Flow edge generation (sequential + branch target jump)."""
        data = asm_to_graph(self.sample_branch_asm, max_len=16)
        ctrl_edges = data[("inst", "control_flow", "inst")].edge_index
        ctrl_pairs = set(zip(ctrl_edges[0].tolist(), ctrl_edges[1].tolist()))

        # Sequential edges (0->1, 1->2, 2->3, 3->4, etc.)
        self.assertIn((0, 1), ctrl_pairs)
        self.assertIn((1, 2), ctrl_pairs)
        self.assertIn((2, 3), ctrl_pairs)

        # Branch target jump edge: blt t0, t1, loop_start (inst 3 -> inst 2 loop_start)
        self.assertIn((3, 2), ctrl_pairs, f"Branch jump edge (3, 2) missing in {ctrl_pairs}")

    def test_def_use_data_flow_edges(self):
        """Test Def -> Use Data Flow edge generation."""
        data = asm_to_graph(self.sample_branch_asm, max_len=16)
        data_edges = data[("inst", "data_flow", "inst")].edge_index
        data_pairs = set(zip(data_edges[0].tolist(), data_edges[1].tolist()))

        # inst 0: li t0, 0 (defines t0)
        # inst 2: addi t0, t0, 1 (uses t0 from inst 0, defines new t0)
        # inst 3: blt t0, t1, loop_start (uses t0 from inst 2)
        # inst 4: mv a0, t0 (uses t0 from inst 2)

        self.assertIn((0, 2), data_pairs, f"Def-use edge (0, 2) missing in {data_pairs}")
        self.assertIn((2, 3), data_pairs, f"Def-use edge (2, 3) missing in {data_pairs}")
        self.assertIn((2, 4), data_pairs, f"Def-use edge (2, 4) missing in {data_pairs}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
