#!/usr/bin/env python3
"""
compile_anghabench_sample.py — Sample 10,000 Functions from AnghaBench and Compile to RISC-V -O0

Features:
- Scans AnghaBench repository for 1,000,000+ C function benchmark files.
- Randomly samples 10,000 C source files.
- Compiles each function to RISC-V 64-bit assembly (-O0) using riscv64-elf-gcc with -std=gnu89 -w -fno-builtin flags.
- Parses generated RISC-V assembly and writes compiled function benchmark dataset to anghabench_10k_corpus.json.
- Integrates seamlessly with SuperoptEnv.
"""

import json
import os
import random
import subprocess
import glob
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Optional

COMPILER = "/usr/bin/riscv64-elf-gcc"
if not os.path.exists(COMPILER):
    COMPILER = "/usr/bin/riscv64-linux-gnu-gcc"

ANGHABENCH_DIR = "AnghaBench"
OUTPUT_CORPUS = "anghabench_10k_corpus.json"
SAMPLE_SIZE = 10000


def compile_single_c_file(c_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Compiles a single C file to RISC-V assembly (-O0) using GNU89 standard and no-builtin flags.
    """
    asm_out_path = c_file_path + ".s"
    cmd = [
        COMPILER,
        "-S",
        "-O0",
        "-w",
        "-std=gnu89",
        "-fno-builtin",
        c_file_path,
        "-o",
        asm_out_path
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and os.path.exists(asm_out_path):
            with open(asm_out_path, "r", errors="ignore") as f:
                asm_content = f.read()

            try:
                os.remove(asm_out_path)
            except OSError:
                pass

            file_id = os.path.basename(c_file_path).replace(".c", "")
            return {
                "id": f"anghabench_{file_id}",
                "name": file_id,
                "source_path": c_file_path,
                "assembly": asm_content
            }
    except Exception:
        pass
    finally:
        if os.path.exists(asm_out_path):
            try:
                os.remove(asm_out_path)
            except OSError:
                pass

    return None


def build_anghabench_10k_corpus(anghabench_dir: str = ANGHABENCH_DIR, sample_size: int = SAMPLE_SIZE):
    print("======================================================================")
    print(f" AnghaBench RISC-V Compilation Pipeline (-O0, Sample: {sample_size})")
    print("======================================================================")
    print(f" Target Compiler : {COMPILER}")
    print(f" Source Directory: {anghabench_dir}")

    print("[*] Discovering C files in AnghaBench repository...")
    all_c_files = []
    for root, _, files in os.walk(anghabench_dir):
        for f in files:
            if f.endswith(".c"):
                all_c_files.append(os.path.join(root, f))

    total_found = len(all_c_files)
    print(f"[*] Discovered {total_found} C files in AnghaBench.")

    if total_found == 0:
        print("[!] Error: No C files found in AnghaBench directory!")
        return

    random.seed(42)
    sample_files = random.sample(all_c_files, min(sample_size, total_found))
    print(f"[*] Sampled {len(sample_files)} C files for RISC-V compilation.")

    print(f"[*] Starting parallel compilation using {os.cpu_count()} CPU workers...")
    start_time = time.time()

    compiled_corpus = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = executor.map(compile_single_c_file, sample_files, chunksize=50)
        for idx, res in enumerate(results, start=1):
            if res is not None:
                compiled_corpus.append(res)
            if idx % 1000 == 0 or idx == len(sample_files):
                print(f"    Progress: {idx:5d}/{len(sample_files)} processed | Compiled Success: {len(compiled_corpus)}")

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n----------------------------------------------------------------------")
    print(f" Writing Compiled Corpus to {OUTPUT_CORPUS}...")
    print("----------------------------------------------------------------------")
    with open(OUTPUT_CORPUS, "w") as f:
        json.dump(compiled_corpus, f, indent=2)

    success_rate = (len(compiled_corpus) / len(sample_files)) * 100.0
    print(f" Successfully Compiled : {len(compiled_corpus)} / {len(sample_files)} ({success_rate:.1f}%)")
    print(f" Compilation Latency   : {elapsed:.2f} seconds ({len(sample_files)/elapsed:.1f} files/sec)")
    print(f" Corpus File Saved     : {OUTPUT_CORPUS} ({os.path.getsize(OUTPUT_CORPUS) / (1024*1024):.2f} MB)")
    print("======================================================================\n")


if __name__ == "__main__":
    build_anghabench_10k_corpus()
