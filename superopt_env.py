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
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env
from reward_env import RewardEnv, ToolchainError
from peephole_rulebook import PeepholeRulebook, OPCODE_MAP, REG_MAP


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
            imm = int(tokens[2], 0)
        except ValueError:
            imm = 0
    elif op_str in ("ADDI", "ANDI") and len(tokens) >= 4:
        rd = REG_MAP.get(tokens[1], 0)
        rs1 = REG_MAP.get(tokens[2], 0)
        try:
            imm = int(tokens[3], 0)
        except ValueError:
            imm = 0
    elif op_str in ("ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR") and len(tokens) >= 4:
        rd = REG_MAP.get(tokens[1], 0)
        rs1 = REG_MAP.get(tokens[2], 0)
        rs2 = REG_MAP.get(tokens[3], 0)
    elif op_str in ("SLLI", "SRLI", "SRAI") and len(tokens) >= 4:
        rd = REG_MAP.get(tokens[1], 0)
        rs1 = REG_MAP.get(tokens[2], 0)
        try:
            imm = int(tokens[3], 0)
        except ValueError:
            imm = 0
    elif op_str == "MV" and len(tokens) >= 3:
        rd = REG_MAP.get(tokens[1], 0)
        rs1 = REG_MAP.get(tokens[2], 0)

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
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.corpus_path = corpus_path
        self.max_len = max_len
        self.num_rules = num_rules
        self.render_mode = render_mode

        # Connect Rulebook
        self.rulebook = PeepholeRulebook(num_rules=self.num_rules)

        self.corpus = self._load_corpus(corpus_path)
        
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

    def _load_corpus(self, path: str) -> List[Dict[str, Any]]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return [
            {
                "id": "func_default_mul",
                "name": "default_mul",
                "assembly": ".section .text\n.globl _start\n_start:\n    li t0, 42\n    li t1, 8\n    mul t2, t0, t1\n    li a0, 0\n    li a7, 93\n    ecall\n"
            }
        ]

    def rule_matches(self, rule_idx: int, obs: Optional[np.ndarray] = None) -> bool:
        if obs is None:
            obs = self._get_obs()
        return self.rulebook.rule_matches(rule_idx, obs)

    def get_action_mask(self, obs: Optional[np.ndarray] = None) -> np.ndarray:
        if obs is None:
            obs = self._get_obs()
        return self.rulebook.get_action_mask(obs)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.max_len, 5), dtype=np.int32)
        for i, inst in enumerate(self.current_program[:self.max_len]):
            obs[i] = np.array(inst, dtype=np.int32)
        return obs.flatten()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0

        idx = self.np_random.integers(0, len(self.corpus))
        self.current_func_meta = self.corpus[idx]

        asm_text = self.current_func_meta["assembly"]
        self.current_program = parse_assembly_program(asm_text, max_len=self.max_len)

        if self.reward_env:
            try:
                self.baseline_cycles = float(self.reward_env.compile_and_run(asm_text))
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

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        rule_id = action // self.max_len
        target_idx = action % self.max_len

        reward = 0.0
        applied = False

        # Delegate rewrite execution to PeepholeRulebook
        if target_idx < len(self.current_program):
            self.current_program, applied = self.rulebook.apply_rewrite(
                rule_id, self.current_program, target_idx
            )

        if applied:
            new_asm_str = program_to_assembly_text(self.current_program)
            if self.reward_env:
                try:
                    new_cycles = float(self.reward_env.compile_and_run(new_asm_str))
                    cycle_ratio = (self.baseline_cycles - new_cycles) / max(1.0, self.baseline_cycles)
                    reward = float(cycle_ratio - 0.01)
                    self.current_cycles = new_cycles
                except Exception:
                    rule_info = self.rulebook.get_rule_info(rule_id)
                    reward = float((rule_info.get("est_savings", 1.0) / max(1.0, self.baseline_cycles)) - 0.01)
            else:
                rule_info = self.rulebook.get_rule_info(rule_id)
                reward = float((rule_info.get("est_savings", 1.0) / max(1.0, self.baseline_cycles)) - 0.01)
        else:
            reward = 0.0

        terminated = applied or (self.step_count >= self.max_steps)
        truncated = False
        obs = self._get_obs()
        
        info = {
            "applied": applied,
            "rule_id": rule_id,
            "target_idx": target_idx,
            "current_cycles": self.current_cycles,
            "baseline_cycles": self.baseline_cycles,
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
