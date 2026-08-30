#!/usr/bin/env python3
"""
superopt_env.py — SuperoptEnv Connected to PeepholeRulebook (N_rules = 10)

Delegates all algebraic pattern matching and rewrite logic to PeepholeRulebook.
100% Zero-Warning Gymnasium Compliant.
"""

import json
import math
import os
import re
import warnings
import collections
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from reward_env import RewardEnv, ToolchainError
from peephole_rulebook import PeepholeRulebook, OPCODE_MAP, REG_MAP
from equivalence_verifier import verify_equiv, verify_equiv_status
from curriculum_scheduler import compute_curriculum_max_len


# ---------------------------------------------------------------------------
# Helper Assembly Converters
# ---------------------------------------------------------------------------

REV_OPCODE_MAP = {v: k for k, v in OPCODE_MAP.items()}
REV_REG_MAP_CANON = {
    0: "zero", 1: "ra", 2: "sp", 3: "gp", 4: "tp",
    5: "t0", 6: "t1", 7: "t2", 8: "s0", 9: "s1",
    10: "a0", 11: "a1", 12: "a2", 13: "a3", 14: "a4", 15: "a5",
    16: "a6", 17: "a7", 18: "s2", 19: "s3", 20: "s4", 21: "s5",
    28: "t3", 29: "t4", 30: "t5", 31: "t6"
}


def rule_matches(rule_idx: int, obs: np.ndarray) -> bool:
    rulebook = PeepholeRulebook(num_rules=10)
    return rulebook.rule_matches(rule_idx, obs)


INT32_MIN = -(2**31)
INT32_MAX = (2**31) - 1

def _clamp_int32(val: int) -> int:
    """Clamp a Python integer to int32 range to prevent numpy overflow."""
    return max(INT32_MIN, min(INT32_MAX, val))

def parse_assembly_instruction(line: str) -> List[int]:
    line = line.strip()
    if not line or line.startswith(".") or line.startswith("#") or line.endswith(":"):
        return [OPCODE_MAP["NOP"], 0, 0, 0, 0]

    if "#" in line:
        line = line.split("#")[0].strip()

    tokens = re.split(r'[\s,]+', line)
    op_str = tokens[0].upper()
    opcode_id = OPCODE_MAP.get(op_str, OPCODE_MAP["OTHER"])

    rd, rs1, rs2, imm = 0, 0, 0, 0

    if op_str == "LI" and len(tokens) >= 3:
        rd = REG_MAP.get(tokens[1], 0)
        try:
            imm = _clamp_int32(int(tokens[2], 0))
        except ValueError:
            imm = 0
    elif op_str in ("ADDI", "SLTI", "SLTIU", "XORI", "ORI", "ANDI", "SLLI", "SLLI_I", "SRLI", "SRAI", "JALR") and len(tokens) >= 3:
        rd = REG_MAP.get(tokens[1], 0)
        if len(tokens) >= 4:
            rs1 = REG_MAP.get(tokens[2], 0)
            try:
                imm = _clamp_int32(int(tokens[3], 0))
            except ValueError:
                imm = 0
        else:
            try:
                imm = _clamp_int32(int(tokens[2], 0))
            except ValueError:
                imm = 0
    elif op_str in ("ADD", "SUB", "SLL", "SLT", "SLTU", "SRL", "SRA", "MUL", "MULH", "MULHSU", "MULHU", "DIV", "DIVU", "REM", "REMU", "AND", "OR", "XOR") and len(tokens) >= 4:
        rd = REG_MAP.get(tokens[1], 0)
        rs1 = REG_MAP.get(tokens[2], 0)
        rs2 = REG_MAP.get(tokens[3], 0)
    elif op_str in ("BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU") and len(tokens) >= 3:
        rs1 = REG_MAP.get(tokens[1], 0)
        rs2 = REG_MAP.get(tokens[2], 0)
    elif op_str in ("JAL", "J") and len(tokens) >= 2:
        if len(tokens) >= 3:
            rd = REG_MAP.get(tokens[1], 0)
            try:
                imm = _clamp_int32(int(tokens[2], 0))
            except ValueError:
                imm = 0
        else:
            try:
                imm = _clamp_int32(int(tokens[1], 0))
            except ValueError:
                imm = 0

    return [opcode_id, rs1, rs2, rd, imm]


def parse_assembly_program(asm_text: str, max_len: int = 16) -> List[List[int]]:
    encoded = []
    for line in asm_text.splitlines():
        parsed = parse_assembly_instruction(line)
        if parsed[0] != OPCODE_MAP["NOP"] or line.strip() == "nop":
            encoded.append(parsed)
            if len(encoded) >= max_len:
                break
    while len(encoded) < max_len:
        encoded.append([OPCODE_MAP["NOP"], 0, 0, 0, 0])
    return encoded


def program_to_assembly_text(program: List[List[int]]) -> str:
    lines = [".section .text", ".globl _start", "_start:"]
    for inst in program:
        opcode_id, rs1, rs2, rd, imm = inst
        op_str = REV_OPCODE_MAP.get(opcode_id, "NOP")

        if op_str == "NOP":
            continue
        elif op_str == "LI":
            rd_str = REV_REG_MAP_CANON.get(rd, "t0")
            lines.append(f"    li {rd_str}, {imm}")
        elif op_str in ("ADDI", "ANDI"):
            rd_str = REV_REG_MAP_CANON.get(rd, "t0")
            rs1_str = REV_REG_MAP_CANON.get(rs1, "t0")
            lines.append(f"    {op_str.lower()} {rd_str}, {rs1_str}, {imm}")
        elif op_str in ("ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR"):
            rd_str = REV_REG_MAP_CANON.get(rd, "t0")
            rs1_str = REV_REG_MAP_CANON.get(rs1, "t0")
            rs2_str = REV_REG_MAP_CANON.get(rs2, "t0")
            lines.append(f"    {op_str.lower()} {rd_str}, {rs1_str}, {rs2_str}")
        elif op_str in ("SLLI", "SRLI", "SRAI"):
            rd_str = REV_REG_MAP_CANON.get(rd, "t0")
            rs1_str = REV_REG_MAP_CANON.get(rs1, "t0")
            lines.append(f"    {op_str.lower()} {rd_str}, {rs1_str}, {imm}")
        elif op_str == "MV":
            rd_str = REV_REG_MAP_CANON.get(rd, "t0")
            rs1_str = REV_REG_MAP_CANON.get(rs1, "t0")
            lines.append(f"    mv {rd_str}, {rs1_str}")
        elif op_str == "ECALL":
            lines.append("    ecall")

    if not any("ecall" in line for line in lines):
        lines.append("    li a0, 0")
        lines.append("    li a7, 93")
        lines.append("    ecall")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# SuperoptEnv Class Definition (Connected to PeepholeRulebook)
# ---------------------------------------------------------------------------

class SuperoptEnv(gym.Env):
    """
    Gymnasium Superoptimization Environment connected to PeepholeRulebook (N_rules = 10).
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        corpus_path: str = "corpus.json",
        max_len: int = 16,
        num_rules: int = 10,
        use_graph_obs: bool = True,
        use_reward_shaping: bool = True,
        gamma: float = 0.99,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.corpus_path = corpus_path
        self.max_len = max_len
        self.num_rules = num_rules
        self.use_graph_obs = use_graph_obs
        self.use_reward_shaping = use_reward_shaping
        self.gamma = gamma
        self.render_mode = render_mode

        # Connect Rulebook
        self.rulebook = PeepholeRulebook(num_rules=self.num_rules)

        self.corpus = self._load_corpus(corpus_path)

        # Verification Status Counters per Rule
        self.rule_sat_counts: Dict[int, int] = collections.defaultdict(int)
        self.rule_unsat_counts: Dict[int, int] = collections.defaultdict(int)
        self.rule_timeout_counts: Dict[int, int] = collections.defaultdict(int)

        try:
            self.reward_env = RewardEnv(strict=False)
        except ToolchainError:
            self.reward_env = None

        # Observation Space: Flat int32 array [opcode, rs1, rs2, rd, imm] x max_len
        self.observation_space = spaces.Box(
            low=-2147483648,
            high=2147483647,
            shape=(self.max_len * 5,),
            dtype=np.int32
        )

        # Action Space: Discrete(num_rules * max_len)
        self.action_space = spaces.Discrete(self.num_rules * self.max_len)

        self.current_func_meta: Dict[str, Any] = {}
        self.current_program: List[List[int]] = []
        self.baseline_cycles: float = 10.0
        self.current_cycles: float = 10.0
        self.step_count = 0
        self.max_steps = 20

    def update_curriculum(self, global_step: int) -> int:
        """
        Dynamically update max_len based on formula: min(8 + floor(17 * step / 500000), 25)
        """
        new_max_len = compute_curriculum_max_len(global_step)
        if new_max_len != self.max_len:
            self.max_len = new_max_len
            self.observation_space = spaces.Box(
                low=-2147483648,
                high=2147483647,
                shape=(self.max_len * 5,),
                dtype=np.int32
            )
            self.action_space = spaces.Discrete(self.num_rules * self.max_len)
        return self.max_len

    def _load_corpus(self, path: str) -> List[Dict[str, Any]]:
        # Auto-fallback to full index corpus_index.jsonl if present and path is default/missing
        if (path == "corpus.json" or not os.path.exists(path)) and os.path.exists("corpus_index.jsonl"):
            path = "corpus_index.jsonl"

        if os.path.exists(path):
            if path.endswith(".jsonl"):
                # JSONL lazy-loading: store only lightweight index entries.
                # Each line is a JSON object with 'asm_path' pointing to a
                # .s file on disk; the assembly text is NOT loaded here.
                corpus: List[Dict[str, Any]] = []
                corpus_dir = os.path.dirname(os.path.abspath(path))
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        # Resolve relative asm_path against corpus directory
                        if "asm_path" in entry and not os.path.isabs(entry["asm_path"]):
                            entry["asm_path"] = os.path.join(corpus_dir, entry["asm_path"])
                        corpus.append(entry)
                return corpus
            else:
                # Standard JSON corpus (list of dicts with inline 'assembly')
                with open(path, "r") as f:
                    return json.load(f)
        return [
            {
                "id": "func_default_mul",
                "name": "default_mul",
                "assembly": ".section .text\n.globl _start\n_start:\n    li t0, 42\n    li t1, 8\n    mul t2, t0, t1\n    li a0, 0\n    li a7, 93\n    ecall\n"
            }
        ]

    def rule_matches(self, rule_idx: int, obs: Optional[Any] = None) -> bool:
        obs_flat = self._get_obs_flat()
        return self.rulebook.rule_matches(rule_idx, obs_flat)

    def get_action_mask(self, obs: Optional[Any] = None) -> np.ndarray:
        obs_flat = self._get_obs_flat()
        return self.rulebook.get_action_mask(obs_flat)

    def _get_obs_flat(self) -> np.ndarray:
        obs = np.zeros((self.max_len, 5), dtype=np.int64)
        for i, inst in enumerate(self.current_program[:self.max_len]):
            obs[i] = np.array(inst, dtype=np.int64)
        obs = np.clip(obs, INT32_MIN, INT32_MAX).astype(np.int32)
        return obs.flatten()

    def _get_obs(self) -> Any:
        if self.use_graph_obs:
            from asm_graph_builder import asm_to_graph
            asm_text = program_to_assembly_text(self.current_program)
            return asm_to_graph(asm_text, max_len=self.max_len)
        return self._get_obs_flat()

    def _count_instructions(self, meta: Dict[str, Any]) -> int:
        if "assembly" in meta and meta["assembly"]:
            lines = [l.strip() for l in meta["assembly"].splitlines() if l.strip() and not l.strip().startswith(".") and not l.strip().startswith("#") and not l.strip().endswith(":")]
            return len(lines)
        return meta.get("length", meta.get("baseline_cycles", 8))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        # Filter corpus: only sample functions with len(instrs) <= current max_len
        valid_indices = [
            i for i, meta in enumerate(self.corpus)
            if self._count_instructions(meta) <= self.max_len
        ]

        if valid_indices:
            idx = int(self.np_random.choice(valid_indices))
        else:
            idx = int(self.np_random.integers(0, len(self.corpus)))

        self.current_func_meta = self.corpus[idx]

        # Lazy-load: read assembly from disk if asm_path is present,
        # otherwise use inline 'assembly' text (backward compatible).
        if "asm_path" in self.current_func_meta:
            asm_file = self.current_func_meta["asm_path"]
            with open(asm_file, "r") as f:
                asm_text = f.read()
        else:
            asm_text = self.current_func_meta["assembly"]
        self.current_program = parse_assembly_program(asm_text, max_len=self.max_len)

        if self.reward_env:
            try:
                baseline = float(self.reward_env.compile_and_run(asm_text))
                # Guard: if compile_and_run returns penalty (negative), use corpus default
                if baseline > 0:
                    self.baseline_cycles = baseline
                else:
                    self.baseline_cycles = float(self.current_func_meta.get("baseline_cycles", 10.0))
            except Exception:
                self.baseline_cycles = float(self.current_func_meta.get("baseline_cycles", 10.0))
        else:
            self.baseline_cycles = float(self.current_func_meta.get("baseline_cycles", 10.0))

        self.current_cycles = self.baseline_cycles
        obs = self._get_obs()
        
        info = {
            "corpus_id": self.current_func_meta.get("id", f"func_{idx}"),
            "func_name": self.current_func_meta.get("name", f"func_{idx}"),
            "baseline_cycles": self.baseline_cycles,
            "action_mask": self.get_action_mask(obs)
        }
        return obs, info

    def _get_potential(self) -> float:
        """
        Potential function Phi(s) = -instruction_count(s).
        Counts non-NOP instructions in self.current_program.
        """
        n_instrs = sum(1 for inst in self.current_program if inst[0] != OPCODE_MAP["NOP"])
        return -float(n_instrs)

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        rule_id = action // self.max_len
        target_idx = action % self.max_len

        # Potential at state s
        phi_s = self._get_potential()

        reward = 0.0
        applied = False

        # Delegate rewrite execution to PeepholeRulebook
        if target_idx < len(self.current_program):
            orig_program = [list(inst) for inst in self.current_program]
            candidate_program, applied = self.rulebook.apply_rewrite(
                rule_id, self.current_program, target_idx
            )
            if applied:
                status = verify_equiv_status(orig_program, candidate_program, timeout=5.0)
                if status == "UNSAT":
                    self.current_program = candidate_program
                    self.rule_unsat_counts[rule_id] += 1
                elif status == "TIMEOUT":
                    applied = False
                    reward = -0.1  # Soft penalty for timeout: softer than incorrect (-1.0), harder than no-op (-0.01)
                    self.rule_timeout_counts[rule_id] += 1
                else:  # SAT or UNKNOWN
                    applied = False
                    reward = -1.0  # Hard penalty for incorrect / non-equivalent rewrite
                    self.rule_sat_counts[rule_id] += 1

        if applied:
            new_asm_str = program_to_assembly_text(self.current_program)
            if self.reward_env:
                try:
                    new_cycles = float(self.reward_env.compile_and_run(new_asm_str))
                    # Guard: if compile_and_run returns penalty (negative), use est_savings
                    if new_cycles < 0:
                        rule_info = self.rulebook.get_rule_info(rule_id)
                        reward = float((rule_info.get("est_savings", 1.0) / max(1.0, self.baseline_cycles)) - 0.01)
                    else:
                        cycle_ratio = (self.baseline_cycles - new_cycles) / max(1.0, self.baseline_cycles)
                        reward = float(cycle_ratio - 0.01)
                        self.current_cycles = new_cycles
                except Exception:
                    rule_info = self.rulebook.get_rule_info(rule_id)
                    reward = float((rule_info.get("est_savings", 1.0) / max(1.0, self.baseline_cycles)) - 0.01)
            else:
                rule_info = self.rulebook.get_rule_info(rule_id)
                reward = float((rule_info.get("est_savings", 1.0) / max(1.0, self.baseline_cycles)) - 0.01)

        # Potential-based reward shaping: r_shaped = r + gamma * Phi(s') - Phi(s)
        phi_s_prime = self._get_potential()
        if self.use_reward_shaping:
            shaping_signal = (self.gamma * phi_s_prime) - phi_s
            reward = float(reward + shaping_signal)

        # Episode termination tuning: success terminal if total_speedup > 15% (bonus +1.0)
        total_speedup = (self.baseline_cycles - self.current_cycles) / max(1.0, self.baseline_cycles)
        is_success = (total_speedup > 0.15)

        if is_success:
            reward = float(reward + 1.0)  # Bonus +1.0 for achieving >15% speedup

        terminated = is_success or applied or (self.step_count >= self.max_steps)
        truncated = False
        obs = self._get_obs()
        
        info = {
            "applied": applied,
            "rule_id": rule_id,
            "target_idx": target_idx,
            "current_cycles": self.current_cycles,
            "baseline_cycles": self.baseline_cycles,
            "total_speedup": total_speedup,
            "success": is_success,
            "action_mask": self.get_action_mask(obs)
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode == "human":
            asm = program_to_assembly_text(self.current_program)
            print(f"\n[SuperoptEnv Render]\n{asm}")
        return None

    def close(self):
        pass


# Register SuperoptEnv in Gymnasium Registry safely
if "SuperoptEnv-v0" not in gym.envs.registry:
    register(
        id="SuperoptEnv-v0",
        entry_point="superopt_env:SuperoptEnv",
        max_episode_steps=20
    )


if __name__ == "__main__":
    print("======================================================================")
    print(" Verifying SuperoptEnv Connected to PeepholeRulebook (N_rules = 10)")
    print("======================================================================")

    env = SuperoptEnv(corpus_path="corpus.json", max_len=16, num_rules=10)
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        registered_env = gym.make("SuperoptEnv-v0")
        check_env(registered_env.unwrapped)
        registered_env.close()

    print(f" Action Space            : {env.action_space} (num_rules=10)")
    print(f" Total Warnings Caught: {len(caught_warnings)}")

    assert len(caught_warnings) == 0, f"Expected 0 warnings from check_env, but got {len(caught_warnings)}!"
    print("======================================================================")
    print("[+] PEEPHOLE RULEBOOK INTEGRATION VERIFIED: 100% Gymnasium Compliant!")
    print("======================================================================\n")
