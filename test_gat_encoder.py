#!/usr/bin/env python3
"""
test_gat_encoder.py — Unit Tests for PyG HeteroConv HeteroGATEncoder
"""

import unittest
import torch
from asm_graph_builder import asm_to_graph
from gat_encoder import HeteroGATEncoder, HAS_PYG


class TestHeteroGATEncoder(unittest.TestCase):

    def setUp(self):
        if not HAS_PYG:
            self.skipTest("PyTorch Geometric is not installed.")

        self.sample_asm = """
        .section .text
        .globl _start
        _start:
            li t0, 42
            li t1, 8
            mul t2, t0, t1
            addi t3, t2, 10
            li a0, 0
            li a7, 93
            ecall
        """

    def test_single_hetero_graph_forward(self):
        """Test HeteroData forward pass through HeteroConv encoder."""
        graph = asm_to_graph(self.sample_asm, max_len=16)
        encoder = HeteroGATEncoder(inst_in_dim=177, reg_in_dim=1, hidden_channels=128, out_channels=256)

        graph_embed, inst_embed = encoder(graph.x_dict, graph.edge_index_dict)

        self.assertEqual(graph_embed.shape, (1, 256))
        self.assertEqual(inst_embed.shape, (16, 128))

    def test_hetero_gradient_flow(self):
        """Test gradient flow across heterogeneous relations."""
        graph = asm_to_graph(self.sample_asm, max_len=16)
        encoder = HeteroGATEncoder(inst_in_dim=177, reg_in_dim=1, hidden_channels=128, out_channels=256)

        graph_embed, _ = encoder(graph.x_dict, graph.edge_index_dict)
        loss = graph_embed.sum()
        loss.backward()

        self.assertIsNotNone(encoder.inst_input_proj[0].weight.grad)
        self.assertIsNotNone(encoder.reg_input_proj[0].weight.grad)
        self.assertIsNotNone(encoder.pool_proj[0].weight.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
