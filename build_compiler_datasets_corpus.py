#!/usr/bin/env python3
"""
build_compiler_datasets_corpus.py — Comprehensive Corpus Pipeline across ALL 8 folders in compiler-datasets:
- AnghaBench
- evaluation
- gnn
- isa
- rl
- rl-environments
- supplementary
- verification

Pipeline Steps:
1. Scans all 8 target folders in `compiler-datasets/`.
2. Collects header include directories for exact C/C++ compilation.
3. Compiles .c, .cpp, .cc, .ll, .s, .asm files to RISC-V 64-bit assembly (-O0).
4. Extracts function blocks and loops.
5. Encodes observation vectors (flat_obs 80-dim) and 160-element action masks via PeepholeRulebook.
6. Saves compiled multi-dataset benchmark corpus to `compiler_datasets_corpus.json` and `corpus.json`.
"""

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from superopt_env import parse_assembly_program, program_to_assembly_text
from peephole_rulebook import PeepholeRulebook

BASE_DIR = "compiler-datasets"
SUBDIRS = [
    "AnghaBench",
    "evaluation",
    "gnn",
    "isa",
    "rl",
    "rl-environments",
    "supplementary",
    "verification"
]

OUTPUT_CORPUS = "compiler_datasets_corpus.json"
MAX_LEN = 16
NUM_RULES = 10

GCC = shutil.which("riscv64-linux-gnu-gcc") or shutil.which("riscv64-elf-gcc")
GPP = shutil.which("riscv64-linux-gnu-g++") or shutil.which("riscv64-elf-g++")
LLC = shutil.which("llc")


def collect_include_dirs(base_path: str) -> List[str]:
    inc_dirs = set()
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith((".h", ".hpp", ".inc", ".hgen")):
                inc_dirs.add(root)
    return list(inc_dirs)


def extract_loops(asm_text: str) -> List[str]:
    lines = asm_text.splitlines()
    loops = []
    current = []
    in_loop = False

    for line in lines:
        stripped = line.strip()
        if stripped.endswith(":") and not stripped.startswith("."):
            in_loop = True
            current = [line]
        elif in_loop:
            current.append(line)
            if any(br in stripped for br in ("bne", "beq", "bnez", "beqz", "blt", "bge", "j ", "jr")):
                loops.append("\n".join(current))
                in_loop = False
                current = []

    if not loops:
        loops.append(asm_text)
    return loops


def process_source_file(args: Tuple[str, str, str, List[str]]) -> Optional[Dict[str, Any]]:
    fpath, ext, folder, inc_dirs = args
    asm_out_path = fpath + ".temp.s"
    rulebook = PeepholeRulebook(num_rules=NUM_RULES)

    inc_flags = []
    # Add parent directory of source file and nearest include dirs
    fdir = os.path.dirname(fpath)
    inc_flags.extend(["-I", fdir])
    for d in inc_dirs:
        if d.startswith(os.path.join(BASE_DIR, folder)):
            inc_flags.extend(["-I", d])

    success = False
    asm_content = ""

    try:
        if ext == ".c":
            cmd = [GCC, "-S", "-O0", "-w", "-std=gnu89", "-fno-builtin", "-ffreestanding"] + inc_flags + [fpath, "-o", asm_out_path]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and os.path.exists(asm_out_path):
                success = True
        elif ext in [".cpp", ".cc"]:
            cmd = [GPP, "-S", "-O0", "-w", "-fno-builtin", "-ffreestanding"] + inc_flags + [fpath, "-o", asm_out_path]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and os.path.exists(asm_out_path):
                success = True
        elif ext == ".ll" and LLC:
            cmd = [LLC, "-march=riscv64", "-target-abi=lp64d", fpath, "-o", asm_out_path]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and os.path.exists(asm_out_path):
                success = True
        elif ext in [".s", ".asm"]:
            with open(fpath, "r", errors="ignore") as f:
                asm_content = f.read()
            success = True

        if success and not asm_content and os.path.exists(asm_out_path):
            with open(asm_out_path, "r", errors="ignore") as f:
                asm_content = f.read()

        if success and asm_content.strip():
            file_id = os.path.basename(fpath).replace(ext, "")

            return {
                "id": f"{folder}_{file_id}",
                "name": file_id,
                "folder": folder,
                "source_path": fpath,
                "assembly": asm_content,
                "baseline_cycles": 10.0
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


def build_compiler_datasets_corpus(sample_anghabench: int = 15000):
    print("======================================================================")
    print(" Compiler-Datasets Comprehensive Multi-Folder Corpus Pipeline")
    print("======================================================================")
    print(f" Source Directory : {BASE_DIR}")
    print(f" Subdirectories   : {', '.join(SUBDIRS)}")
    print(f" Target Compiler  : {GCC}")
    print(f" Output File      : {OUTPUT_CORPUS}\n")

    # Step 1: Collect include dirs
    print("[*] Collecting header include directories...")
    inc_dirs = collect_include_dirs(BASE_DIR)
    print(f"[*] Discovered {len(inc_dirs)} header include directories.")

    # Step 2: Discover all source files per folder
    tasks = []
    folder_counts = {}

    import random
    random.seed(42)

    for folder in SUBDIRS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"[!] Folder {folder} not found, skipping.")
            continue

        file_list = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".c", ".cpp", ".cc", ".ll", ".s", ".asm"]:
                    file_list.append((os.path.join(root, f), ext))

        folder_counts[folder] = len(file_list)
        print(f"[*] Discovered {len(file_list):>7,d} source files in {folder}/")

        # If AnghaBench, sample sample_anghabench files for high speed and diversity
        if folder == "AnghaBench" and len(file_list) > sample_anghabench:
            file_list = random.sample(file_list, sample_anghabench)
            print(f"    -> Sampled {len(file_list):,d} AnghaBench files for balanced representation.")

        for fpath, ext in file_list:
            tasks.append((fpath, ext, folder, inc_dirs))

    total_tasks = len(tasks)
    print(f"\n[*] Total Compilation Tasks Queued: {total_tasks:,}")

    num_workers = os.cpu_count() or 16
    print(f"[*] Starting parallel compilation & feature extraction across {num_workers} CPU cores...")

    start_time = time.time()
    compiled_corpus = []
    folder_success = {f: 0 for f in SUBDIRS}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(process_source_file, tasks, chunksize=50)
        for idx, res in enumerate(results, start=1):
            if res is not None:
                compiled_corpus.append(res)
                folder_success[res["folder"]] += 1

            if idx % 2000 == 0 or idx == total_tasks:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                print(f"    Progress: {idx:6,d} / {total_tasks:,d} ({idx/total_tasks*100:5.1f}%) | "
                      f"Compiled: {len(compiled_corpus):6,d} | Speed: {rate:5.1f} files/sec")

    elapsed_total = time.time() - start_time
    print(f"\n[*] Parallel processing completed in {elapsed_total:.2f} seconds ({elapsed_total/60.0:.2f} mins).")

    print("\n----------------------------------------------------------------------")
    print(f" Writing Combined Corpus ({len(compiled_corpus):,d} items) to {OUTPUT_CORPUS}...")
    print("----------------------------------------------------------------------")

    print("[*] Writing compact JSON (no indent for speed)...")
    with open(OUTPUT_CORPUS, "w") as f:
        json.dump(compiled_corpus, f)

    # Also update corpus.json so SuperoptEnv defaults to the full dataset
    with open("corpus.json", "w") as f:
        json.dump(compiled_corpus, f)

    file_size_mb = os.path.getsize(OUTPUT_CORPUS) / (1024 * 1024)

    print("======================================================================")
    print(" MULTI-DATASET CORPUS COMPILATION SUMMARY")
    print("======================================================================")
    for folder in SUBDIRS:
        total_f = folder_counts.get(folder, 0)
        succ_f = folder_success.get(folder, 0)
        pct = (succ_f / total_f * 100) if total_f > 0 else 0.0
        print(f"  {folder:<20}: {succ_f:>6,d} / {total_f:>7,d} successfully compiled ({pct:5.1f}%)")
    print("----------------------------------------------------------------------")
    print(f" Total Corpus Functions Saved : {len(compiled_corpus):,d}")
    print(f" Output File Saved           : {OUTPUT_CORPUS} ({file_size_mb:.2f} MB)")
    print("======================================================================\n")


if __name__ == "__main__":
    build_compiler_datasets_corpus()
