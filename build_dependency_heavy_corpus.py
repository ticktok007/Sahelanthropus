#!/usr/bin/env python3
"""
build_dependency_heavy_corpus.py — Extract assembly functions with max def-use chain >= 3
(including matrix-multiply loops) and construct a reaching-definitions supervised dataset.
"""

import json
import os
import collections
from typing import List, Dict, Tuple, Set
import torch

from asm_graph_builder import asm_to_graph, parse_assembly_with_labels, HAS_PYG
from superopt_env import parse_assembly_program


def count_max_def_use_chain(graph) -> Tuple[int, int]:
    """Compute (num_data_flow_edges, max_def_use_chain_length) for a PyG graph."""
    edge_index = graph['inst', 'data_flow', 'inst'].edge_index
    if edge_index.shape[1] == 0:
        return 0, 0
    num_edges = edge_index.shape[1]

    adj = collections.defaultdict(list)
    in_degree = collections.defaultdict(int)
    nodes = set()

    for u, v in edge_index.t().tolist():
        adj[u].append(v)
        in_degree[v] += 1
        nodes.add(u)
        nodes.add(v)

    dist = {n: 1 for n in nodes}
    queue = collections.deque([n for n in nodes if in_degree[n] == 0])
    max_chain = 1

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            dist[v] = max(dist[v], dist[u] + 1)
            max_chain = max(max_chain, dist[v])
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return num_edges, max_chain


def compute_reaching_definitions(asm_text: str, max_len: int = 16) -> torch.Tensor:
    """
    Compute binary reaching-definition targets for instructions in asm_text.
    For each instruction i (0..N-1), target is a 32-dim binary vector indicating
    which register definitions (1..31) reach instruction i.
    """
    prog, raw_lines, _ = parse_assembly_with_labels(asm_text, max_len=max_len)
    num_insts = len(prog)
    reaching = torch.zeros((num_insts, 32), dtype=torch.float32)

    last_def: Dict[int, int] = {}
    for i, inst in enumerate(prog):
        opcode_id, rs1, rs2, rd, imm = inst
        # Registers that are read at step i receive reaching definitions
        for reg in (rs1, rs2):
            if reg > 0 and reg in last_def:
                reaching[i, reg] = 1.0

        if rd > 0:
            last_def[rd] = i

    return reaching


def main():
    print("=" * 70)
    print(" BUILDING DEPENDENCY-HEAVY CORPUS & REACHING-DEFINITIONS DATASET")
    print("=" * 70)

    heavy_corpus = []
    reaching_defs_data = []

    # 1. Parse matrix-multiply assembly from test_gemm.s
    if os.path.exists("test_gemm.s"):
        with open("test_gemm.s", "r") as f:
            gemm_text = f.read()

        # Split into blocks / subroutines
        blocks = gemm_text.split("\n\n")
        gemm_count = 0
        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip() and not l.strip().startswith(".")]
            if len(lines) >= 4:
                asm_str = "\n".join(lines[:16])
                try:
                    g = asm_to_graph(asm_str)
                    num_edges, max_chain = count_max_def_use_chain(g)
                    if max_chain >= 3:
                        target = compute_reaching_definitions(asm_str)
                        heavy_corpus.append({
                            "name": f"gemm_block_{gemm_count}",
                            "assembly": asm_str,
                            "baseline_cycles": len(lines),
                            "max_chain": max_chain
                        })
                        reaching_defs_data.append((g, target))
                        gemm_count += 1
                except Exception:
                    pass
        print(f" [+] Extracted {gemm_count} matrix-multiply inner loop blocks from test_gemm.s")

    # 2. Filter corpus.json for max def-use chain >= 3
    if os.path.exists("corpus.json"):
        with open("corpus.json", "r") as f:
            corpus = json.load(f)

        print(f" [+] Scanning {len(corpus):,d} functions in corpus.json...")
        corpus_added = 0
        for item in corpus:
            asm_str = item.get("assembly", "")
            if not asm_str:
                continue
            try:
                g = asm_to_graph(asm_str)
                num_edges, max_chain = count_max_def_use_chain(g)
                if max_chain >= 3:
                    target = compute_reaching_definitions(asm_str)
                    item_copy = dict(item)
                    item_copy["max_chain"] = max_chain
                    heavy_corpus.append(item_copy)
                    reaching_defs_data.append((g, target))
                    corpus_added += 1
                    if len(heavy_corpus) >= 500:
                        break
            except Exception:
                pass
        print(f" [+] Extracted {corpus_added} dependency-heavy functions from corpus.json")

    # Save outputs
    output_corpus_path = "dependency_heavy_corpus.json"
    with open(output_corpus_path, "w") as f:
        json.dump(heavy_corpus, f, indent=2)

    output_dataset_path = "reaching_defs_dataset.pt"
    torch.save(reaching_defs_data, output_dataset_path)

    print(f"\n[+] Saved {len(heavy_corpus)} dependency-heavy functions to {output_corpus_path}")
    print(f"[+] Saved {len(reaching_defs_data)} graph reaching-def targets to {output_dataset_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
