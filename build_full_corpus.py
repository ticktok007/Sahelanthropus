#!/usr/bin/env python3
"""
build_full_corpus.py — Full Corpus Pipeline: ALL files from ALL 8 compiler-datasets folders.

Key differences from the sampled version:
  - Compiles ALL 1,044,021 AnghaBench files (no sampling)
  - Saves compiled assembly as individual .s files on disk (memory-efficient)
  - Writes a lightweight JSONL index (one entry per line, ~100 MB)
  - Streaming writes — never holds all assembly text in memory
"""

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple

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

OUTPUT_ASM_DIR = "compiled_asm"
OUTPUT_INDEX = "corpus_index.jsonl"

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


def compile_source_file(args: Tuple[str, str, str, List[str], str]) -> Optional[Dict[str, Any]]:
    """Compile a source file to RISC-V assembly and save to disk."""
    fpath, ext, folder, inc_dirs, asm_out_dir = args
    temp_asm = fpath + ".temp.s"

    # Compute target path first for resume check
    file_id = os.path.basename(fpath).replace(ext, "")
    path_hash = abs(hash(fpath)) % (10**8)
    asm_filename = f"{file_id}_{path_hash}.s"
    asm_out_path = os.path.join(asm_out_dir, folder, asm_filename)

    # Resume: skip if already compiled
    if os.path.exists(asm_out_path) and os.path.getsize(asm_out_path) > 0:
        return {
            "id": f"{folder}_{file_id}_{path_hash}",
            "name": file_id,
            "folder": folder,
            "asm_path": asm_out_path,
            "baseline_cycles": 10.0
        }

    # Build include flags for this folder
    inc_flags = ["-I", os.path.dirname(fpath)]
    for d in inc_dirs:
        if d.startswith(os.path.join(BASE_DIR, folder)):
            inc_flags.extend(["-I", d])

    success = False
    asm_content = ""

    try:
        if ext == ".c":
            cmd = [GCC, "-S", "-O0", "-w", "-std=gnu89", "-fno-builtin", "-ffreestanding"] + inc_flags + [fpath, "-o", temp_asm]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and os.path.exists(temp_asm):
                success = True
        elif ext in [".cpp", ".cc"]:
            cmd = [GPP, "-S", "-O0", "-w", "-fno-builtin", "-ffreestanding"] + inc_flags + [fpath, "-o", temp_asm]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and os.path.exists(temp_asm):
                success = True
        elif ext == ".ll" and LLC:
            cmd = [LLC, "-march=riscv64", "-target-abi=lp64d", fpath, "-o", temp_asm]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and os.path.exists(temp_asm):
                success = True
        elif ext in [".s", ".asm"]:
            with open(fpath, "r", errors="ignore") as f:
                asm_content = f.read()
            success = True

        if success and not asm_content and os.path.exists(temp_asm):
            with open(temp_asm, "r", errors="ignore") as f:
                asm_content = f.read()

        if success and asm_content.strip():
            # Save assembly to disk
            os.makedirs(os.path.dirname(asm_out_path), exist_ok=True)
            with open(asm_out_path, "w") as f:
                f.write(asm_content)

            return {
                "id": f"{folder}_{file_id}_{path_hash}",
                "name": file_id,
                "folder": folder,
                "asm_path": asm_out_path,
                "baseline_cycles": 10.0
            }
    except Exception:
        pass
    finally:
        if os.path.exists(temp_asm):
            try:
                os.remove(temp_asm)
            except OSError:
                pass

    return None


def build_full_corpus():
    print("=" * 70)
    print(" FULL Compiler-Datasets Corpus Pipeline (ALL files, no sampling)")
    print("=" * 70)
    print(f" Source Directory : {BASE_DIR}")
    print(f" Subdirectories  : {', '.join(SUBDIRS)}")
    print(f" Target Compiler : {GCC}")
    print(f" Assembly Output : {OUTPUT_ASM_DIR}/")
    print(f" Index Output    : {OUTPUT_INDEX}")
    print()

    # Step 1: Collect include dirs
    print("[*] Collecting header include directories...")
    inc_dirs = collect_include_dirs(BASE_DIR)
    print(f"[*] Discovered {len(inc_dirs)} header include directories.")

    # Step 2: Create output directories
    os.makedirs(OUTPUT_ASM_DIR, exist_ok=True)
    for folder in SUBDIRS:
        os.makedirs(os.path.join(OUTPUT_ASM_DIR, folder), exist_ok=True)

    # Step 3: Discover ALL source files (no sampling!)
    tasks = []
    folder_counts = {}

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
        print(f"[*] Discovered {len(file_list):>10,d} source files in {folder}/")

        for fpath, ext in file_list:
            tasks.append((fpath, ext, folder, inc_dirs, OUTPUT_ASM_DIR))

    total_tasks = len(tasks)
    print(f"\n[*] Total Compilation Tasks: {total_tasks:,} (NO SAMPLING)")

    num_workers = os.cpu_count() or 16
    print(f"[*] Starting parallel compilation across {num_workers} CPU cores...")
    print(f"[*] Estimated time: ~{total_tasks / 350 / 60:.0f} minutes\n")

    start_time = time.time()
    compiled_count = 0
    folder_success = {f: 0 for f in SUBDIRS}

    # Stream results to JSONL — never hold all entries in memory
    with open(OUTPUT_INDEX, "w") as index_file:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(compile_source_file, tasks, chunksize=200)
            for idx, res in enumerate(results, start=1):
                if res is not None:
                    # Write one JSON line per compiled file
                    index_file.write(json.dumps(res) + "\n")
                    compiled_count += 1
                    folder_success[res["folder"]] += 1

                if idx % 10000 == 0 or idx == total_tasks:
                    elapsed = time.time() - start_time
                    rate = idx / elapsed
                    eta_min = (total_tasks - idx) / rate / 60 if rate > 0 else 0
                    print(f"    Progress: {idx:>10,d} / {total_tasks:,d} ({idx/total_tasks*100:5.1f}%) | "
                          f"Compiled: {compiled_count:>10,d} | "
                          f"Speed: {rate:5.0f} files/sec | "
                          f"ETA: {eta_min:.1f} min")

    elapsed_total = time.time() - start_time

    # Also update corpus.json for backward compat (lightweight — just paths)
    # We'll copy the index to corpus_index.jsonl which SuperoptEnv reads

    index_size_mb = os.path.getsize(OUTPUT_INDEX) / (1024 * 1024)
    asm_dir_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(OUTPUT_ASM_DIR)
        for f in fns
    ) / (1024 * 1024)

    print(f"\n{'=' * 70}")
    print(f" FULL CORPUS COMPILATION SUMMARY")
    print(f"{'=' * 70}")
    print(f" Duration: {elapsed_total:.0f} sec ({elapsed_total/60:.1f} min)")
    print()
    for folder in SUBDIRS:
        total_f = folder_counts.get(folder, 0)
        succ_f = folder_success.get(folder, 0)
        pct = (succ_f / total_f * 100) if total_f > 0 else 0.0
        print(f"  {folder:<20}: {succ_f:>10,d} / {total_f:>10,d} ({pct:5.1f}%)")
    print(f"-" * 70)
    print(f" Total Functions Compiled : {compiled_count:,}")
    print(f" Index File              : {OUTPUT_INDEX} ({index_size_mb:.1f} MB)")
    print(f" Assembly Files          : {OUTPUT_ASM_DIR}/ ({asm_dir_size:.0f} MB)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    build_full_corpus()
