#!/usr/bin/env python3
"""
precompute_corpus.py — Pre-compute QEMU baseline cycle counts for extracted RISC-V functions and save to corpus.json.
"""

import json
import sys
from pathlib import Path
from reward_env import RewardEnv, ToolchainError

# Benchmark Corpus of extracted RISC-V assembly functions
CORPUS_FUNCTIONS = [
    {
        "id": "func_exit_minimal",
        "name": "exit_minimal",
        "description": "Minimal syscall exit program",
        "assembly": """\
.section .text
.globl _start

_start:
    li a0, 0
    li a7, 93
    ecall
"""
    },
    {
        "id": "func_arithmetic_chain",
        "name": "arithmetic_chain",
        "description": "Chain of 10 ALU arithmetic instructions",
        "assembly": """\
.section .text
.globl _start

_start:
    li t0, 1
    addi t0, t0, 2
    add t1, t0, t0
    sub t2, t1, t0
    and t3, t2, t1
    or  t4, t3, t2
    xor t5, t4, t3
    slli t6, t5, 2
    srli a1, t6, 1
    srai a2, a1, 1

    li a0, 0
    li a7, 93
    ecall
"""
    },
    {
        "id": "func_simple_loop_10",
        "name": "simple_loop_10",
        "description": "Loop executing 10 iterations",
        "assembly": """\
.section .text
.globl _start

_start:
    li t0, 10
.loop:
    addi t0, t0, -1
    bnez t0, .loop

    li a0, 0
    li a7, 93
    ecall
"""
    },
    {
        "id": "func_simple_loop_100",
        "name": "simple_loop_100",
        "description": "Loop executing 100 iterations",
        "assembly": """\
.section .text
.globl _start

_start:
    li t0, 100
.loop:
    addi t0, t0, -1
    bnez t0, .loop

    li a0, 0
    li a7, 93
    ecall
"""
    },
    {
        "id": "func_bitwise_and_shifts",
        "name": "bitwise_and_shifts",
        "description": "Bitwise operations and logical shifts",
        "assembly": """\
.section .text
.globl _start

_start:
    li t0, 0x12345678
    li t1, 0x0F0F0F0F
    and t2, t0, t1
    or  t3, t0, t1
    xor t4, t0, t1
    slli t5, t2, 4
    srli t6, t3, 4

    li a0, 0
    li a7, 93
    ecall
"""
    },
    {
        "id": "func_array_sum",
        "name": "array_sum",
        "description": "Computes sum of a 5-element array on stack",
        "assembly": """\
.section .text
.globl _start

_start:
    addi sp, sp, -32
    li t0, 10
    sd t0, 0(sp)
    li t0, 20
    sd t0, 8(sp)
    li t0, 30
    sd t0, 16(sp)
    li t0, 40
    sd t0, 24(sp)

    ld t1, 0(sp)
    ld t2, 8(sp)
    add t3, t1, t2
    ld t4, 16(sp)
    add t3, t3, t4
    ld t5, 24(sp)
    add t3, t3, t5

    addi sp, sp, 32
    li a0, 0
    li a7, 93
    ecall
"""
    }
]


def precompute_baselines(output_file: str = "corpus.json"):
    print("[+] Initializing RewardEnv for QEMU baseline cycle computation...", file=sys.stderr)
    try:
        env = RewardEnv(strict=True)
    except ToolchainError as exc:
        print(f"[-] Error: Toolchain unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[+] QEMU Binary: {env.qemu_binary}")
    print(f"[+] Plugin Path: {env.plugin_path}\n")

    corpus_data = []

    for item in CORPUS_FUNCTIONS:
        func_id = item["id"]
        func_name = item["name"]
        asm_code = item["assembly"]
        desc = item["description"]

        print(f"-> Pre-computing cycles for '{func_id}' ({func_name})...")
        try:
            baseline_cycles = env.compile_and_run(asm_code)
            
            # Count instructions in assembly
            inst_lines = [
                line.strip() for line in asm_code.splitlines()
                if line.strip() and not line.strip().startswith(".") and not line.strip().endswith(":")
            ]
            inst_count = len(inst_lines)

            entry = {
                "id": func_id,
                "name": func_name,
                "description": desc,
                "instruction_count": inst_count,
                "baseline_cycles": float(baseline_cycles),
                "baseline_score": -float(baseline_cycles),
                "assembly": asm_code
            }

            corpus_data.append(entry)
            print(f"   Baseline Cycles: {baseline_cycles} cycles (Score: {-float(baseline_cycles)})\n")

        except Exception as exc:
            print(f"   [-] Error computing baseline for '{func_id}': {exc}\n", file=sys.stderr)

    # Save to corpus.json
    output_path = Path(output_file).resolve()
    with open(output_path, "w") as f:
        json.dump(corpus_data, f, indent=2)

    print(f"[+] Successfully pre-computed {len(corpus_data)} function baselines -> {output_path}")


if __name__ == "__main__":
    precompute_baselines()
