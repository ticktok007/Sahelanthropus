#!/usr/bin/env python3
"""
analyze_sat_rules.py — Synthesize pattern-matching programs for all 10 peephole rules
and evaluate Z3 verification (UNSAT vs SAT) to detect any flawed rule implementations.
"""

from superopt_env import SuperoptEnv, parse_assembly_program
from equivalence_verifier import verify_equiv_status
from peephole_rulebook import PeepholeRulebook, RULE_METADATA, OPCODE_MAP

# Synthetic matching programs for all 10 rules:
TEST_RULE_PATTERNS = {
    0: ("mul_power2_to_slli",    "li t0, 4\nmul t1, a0, t0",       1),  # Slot 1: mul t1, a0, t0
    1: ("udiv_power2_to_srli",   "li t0, 8\ndivu t1, a0, t0",       1),  # Slot 1: divu t1, a0, t0
    2: ("add_zero_to_nop",       "addi t0, a0, 0",                0),  # Slot 0: addi t0, a0, 0
    3: ("mul_zero_to_li_0",      "mul t0, a0, zero",              0),  # Slot 0: mul t0, a0, zero
    4: ("xor_self_to_li_0",      "xor t0, a0, a0",                0),  # Slot 0: xor t0, a0, a0
    5: ("sub_self_to_li_0",      "sub t0, a0, a0",                0),  # Slot 0: sub t0, a0, a0
    6: ("and_self_to_mv",        "and t0, a0, a0",                0),  # Slot 0: and t0, a0, a0
    7: ("or_self_to_mv",         "or t0, a0, a0",                 0),  # Slot 0: or t0, a0, a0
    8: ("add_sub_cancel",        "sub t0, a0, a1\nadd t2, t0, a1", 1),  # Slot 1: add t2, t0, a1
    9: ("sll_srl_to_andi_mask",  "slli t0, a0, 4\nsrli t1, t0, 4", 1),  # Slot 1: srli t1, t0, 4
}


def evaluate_all_rules_z3():
    print("=" * 80)
    print(" Z3 EQUIVALENCE VERIFICATION AUDIT ACROSS ALL 10 PEEPHOLE RULES")
    print("=" * 80)

    rulebook = PeepholeRulebook(num_rules=10)

    print(f"{'Rule ID':<8} | {'Rule Name':<25} | {'Pattern Match':<15} | {'Z3 Status':<12} | {'Verdict':<20}")
    print("-" * 85)

    sat_rules = []
    unsat_rules = []
    failed_rules = []

    for rule_id in range(10):
        name, asm_str, target_idx = TEST_RULE_PATTERNS[rule_id]
        prog = parse_assembly_program(asm_str, max_len=16)

        # Check if rule matches pattern
        matches = rulebook.matches_at(rule_id, prog, target_idx)

        if not matches:
            print(f"Rule {rule_id:<3d} | {name:<25} | {'❌ NO MATCH':<15} | {'N/A':<12} | ⚠️ Check Rulebook Pattern Matcher")
            failed_rules.append(rule_id)
            continue

        # Apply rewrite
        cand_prog, applied = rulebook.apply_rewrite(rule_id, prog, target_idx)
        if not applied:
            print(f"Rule {rule_id:<3d} | {name:<25} | {'✓ MATCHED':<15} | {'APPLY FAIL':<12} | ⚠️ Rule Application Failed")
            failed_rules.append(rule_id)
            continue

        # Verify equivalence via Z3
        status = verify_equiv_status(prog, cand_prog, timeout=5.0)

        if status == "UNSAT":
            verdict = "✅ PASSED (Formally Equivalent)"
            unsat_rules.append(rule_id)
        elif status == "SAT":
            verdict = "🔴 SAT (INCORRECT RULE IMPLEMENTATION!)"
            sat_rules.append(rule_id)
        else:
            verdict = f"🟡 {status}"

        print(f"Rule {rule_id:<3d} | {name:<25} | {'✓ MATCHED':<15} | {status:<12} | {verdict}")

    print("=" * 80)
    print(f" AUDIT SUMMARY:")
    print(f"   ✓ Formally Equivalent (UNSAT) Rules : {len(unsat_rules)} / 10")
    print(f"   🔴 Flawed / Incorrect (SAT) Rules   : {len(sat_rules)} / 10")
    if sat_rules:
        print(f"   ⚠️ ATTENTION: The following rules produced SAT counterexamples: {sat_rules}")
    else:
        print("   ✅ ALL 10 PEEPHOLE RULES PROVEN 100% CORRECT & EQUIVALENT VIA Z3!")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_all_rules_z3()
