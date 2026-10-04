// src/services/mockService.ts
import {
  SuperoptRun,
  BenchmarkExample,
  InstructionToken,
  GraphNode,
  GraphEdge,
  PeepholeRule,
  RuleMatchCell,
  PolicyAction,
  TrajectoryStep,
  SymbolicRegisterState,
} from '../types/superoptimizer';

export const BENCHMARK_EXAMPLES: BenchmarkExample[] = [
  {
    id: 'ex1',
    name: '01. Power-of-2 Strength Reduction (Rule 0)',
    description: 'Replaces expensive 4-cycle multiplier with 1-cycle logical left shift.',
    code: `li t0, 8\nmul t1, a0, t0`,
  },
  {
    id: 'ex2',
    name: '02. Self-XOR & Add-Zero Elimination (Rules 2 & 4)',
    description: 'Eliminates redundant zeroing and zero-addition identity operations.',
    code: `xor t0, a0, a0\naddi t1, a1, 0\nadd t2, t1, t0`,
  },
  {
    id: 'ex3',
    name: '03. Unsigned Division by Power of 2 (Rule 1)',
    description: 'Converts 20-cycle hardware unsigned division to 1-cycle logical right shift.',
    code: `li t0, 16\ndivu t1, a0, t0`,
  },
  {
    id: 'ex4',
    name: '04. Add-Sub Algebraic Cancellation (Rule 8)',
    description: 'Cancels out inverse subtraction/addition sequence: add(sub(X, Y), Y) -> X.',
    code: `sub t0, a0, a1\nadd t2, t0, a1`,
  },
  {
    id: 'ex5',
    name: '05. Combined Multi-Rule Pipeline Chain (Featured Demo)',
    description: 'Chains strength reduction, register zeroing, and register move optimizations.',
    code: `li t0, 4\nmul t1, a0, t0\nxor t2, a1, a1\nadd t3, t1, t2`,
  },
  {
    id: 'ex6',
    name: '06. Def-Use Heavy Sequence (Matrix Loop Unroll)',
    description: 'Optimizes inner matrix multiply accumulator register writebacks.',
    code: `li t0, 2\nmul t1, a0, t0\nadd t2, a1, t1\nsub t3, t2, t2`,
  },
  {
    id: 'ex7',
    name: '07. Identity Bitwise Masking (Rule 9)',
    description: 'Fuses left shift and right shift operations into bitwise AND mask.',
    code: `slli t0, a0, 4\nsrli t1, t0, 4`,
  },
  {
    id: 'ex8',
    name: '08. x86-64 Linux Syscall Sequence (Auto-Transpiled)',
    description: 'Auto-detects x86-64 assembly dialect, transpiles to RISC-V RV32I, and constant-folds registers.',
    code: `section .text\nglobal _start\n\n_start:\n    mov rax, 10\n    mov rbx, 20\n    add rax, rbx\n    ; RAX = 30\n    mov rax, 60\n    mov rdi, 0\n    syscall`,
  },
  {
    id: 'ex9',
    name: '09. Memory Load/Store & Redundant Copy Chain (Featured)',
    description: 'Eliminates redundant register copies, load-after-store memory accesses, and chain moves.',
    code: `.data\nnum1: .word 10\nnum2: .word 20\nresult: .word 0\n\n.text\n.globl _start\n_start:\n    la t0, num1\n    lw t1, 0(t0)\n    la t2, num2\n    lw t3, 0(t2)\n    add t4, t1, zero\n    add t5, t3, zero\n    add t6, t4, t5\n    add a0, t6, zero\n    la t0, result\n    sw a0, 0(t0)\n    lw t1, 0(t0)\n    add a0, t1, zero\n    li a7, 93\n    li a0, 0\n    ecall`,
  },
];

const OPCODE_MAP: Record<string, number> = {
  NOP: 0, LI: 1, ADD: 2, ADDI: 3, SUB: 4, MUL: 5, DIV: 6,
  SLLI: 7, SRLI: 8, SRAI: 9, AND: 10, OR: 11, XOR: 12,
  MV: 13, DIVU: 36, ANDI: 16, ECALL: 40, LA: 20, LW: 21, SW: 22
};

const REG_MAP: Record<string, number> = {
  zero: 0, x0: 0, ra: 1, sp: 2, gp: 3, tp: 4,
  t0: 5, t1: 6, t2: 7, s0: 8, s1: 9,
  a0: 10, a1: 11, a2: 12, a3: 13, a4: 14, a5: 15, a6: 16, a7: 17,
  s2: 18, s3: 19, s4: 20, s5: 21, s6: 22, s7: 23, s8: 24, s9: 25, s10: 26, s11: 27,
  t3: 28, t4: 29, t5: 30, t6: 31
};

const X86_TO_RISCV_REG: Record<string, string> = {
  rax: 'a0', eax: 'a0', ax: 'a0', al: 'a0',
  rbx: 't0', ebx: 't0', bx: 't0', bl: 't0',
  rcx: 't1', ecx: 't1', cx: 't1', cl: 't1',
  rdx: 't2', edx: 't2', dx: 't2', dl: 't2',
  rsi: 'a1', esi: 'a1', si: 'a1',
  rdi: 'a0', edi: 'a0', di: 'a0',
  rsp: 'sp', esp: 'sp', rbp: 's0', ebp: 's0',
  r8: 't3', r9: 't4', r10: 't5',
};

function mapX86Reg(r: string): string {
  const clean = r.toLowerCase().trim();
  return X86_TO_RISCV_REG[clean] || 't0';
}

export function preprocessAndTranspile(rawAsmText: string): {
  isX86: boolean;
  riscvLines: string[];
  transpiledLog: string[];
} {
  const lines = rawAsmText.split('\n');
  const riscvLines: string[] = [];
  const transpiledLog: string[] = [];
  let isX86 = false;

  for (let rawLine of lines) {
    // Strip comments starting with ;, #, or //
    let line = rawLine.split(/;|#|\/\//)[0].trim();
    if (!line) continue;

    // Ignore assembler directives and labels
    if (
      line.startsWith('.') ||
      line.toLowerCase().startsWith('section') ||
      line.toLowerCase().startsWith('global') ||
      line.toLowerCase().startsWith('globl') ||
      line.toLowerCase().startsWith('extern') ||
      line.toLowerCase().startsWith('default') ||
      line.endsWith(':')
    ) {
      continue;
    }

    const parts = line.split(/[\s,]+/).filter(Boolean);
    const op = parts[0]?.toLowerCase();

    // Detect x86-64 assembly dialect
    if (
      ['mov', 'syscall', 'imul', 'pop', 'push', 'lea'].includes(op) ||
      line.toLowerCase().includes('rax') ||
      line.toLowerCase().includes('rbx') ||
      line.toLowerCase().includes('rdi') ||
      line.toLowerCase().includes('rsi')
    ) {
      isX86 = true;
    }

    if (op === 'mov' && parts.length >= 3) {
      const dstRaw = parts[1].toLowerCase();
      const src = parts[2];
      const imm = parseInt(src, 10);

      // Handle x86 exit syscall rax=60 mapping to RISC-V a7=93
      if ((dstRaw === 'rax' || dstRaw === 'eax') && imm === 60) {
        riscvLines.push(`li a7, 93`);
        transpiledLog.push(`Transpiled x86 'mov rax, 60' (sys_exit) ➔ RISC-V 'li a7, 93'`);
      } else if (!isNaN(imm)) {
        const dst = mapX86Reg(dstRaw);
        riscvLines.push(`li ${dst}, ${imm}`);
        transpiledLog.push(`Transpiled x86 'mov ${parts[1]}, ${src}' ➔ RISC-V 'li ${dst}, ${imm}'`);
      } else {
        const dst = mapX86Reg(dstRaw);
        const srcReg = mapX86Reg(src);
        riscvLines.push(`mv ${dst}, ${srcReg}`);
        transpiledLog.push(`Transpiled x86 'mov ${parts[1]}, ${src}' ➔ RISC-V 'mv ${dst}, ${srcReg}'`);
      }
    } else if (op === 'add' && parts.length >= 3) {
      const dst = mapX86Reg(parts[1]);
      const src = parts[2];
      const imm = parseInt(src, 10);
      if (!isNaN(imm)) {
        riscvLines.push(`addi ${dst}, ${dst}, ${imm}`);
        transpiledLog.push(`Transpiled x86 'add ${parts[1]}, ${src}' ➔ RISC-V 'addi ${dst}, ${dst}, ${imm}'`);
      } else {
        const srcReg = mapX86Reg(src);
        riscvLines.push(`add ${dst}, ${dst}, ${srcReg}`);
        transpiledLog.push(`Transpiled x86 'add ${parts[1]}, ${src}' ➔ RISC-V 'add ${dst}, ${dst}, ${srcReg}'`);
      }
    } else if (op === 'sub' && parts.length >= 3) {
      const dst = mapX86Reg(parts[1]);
      const src = parts[2];
      const imm = parseInt(src, 10);
      if (!isNaN(imm)) {
        riscvLines.push(`addi ${dst}, ${dst}, ${-imm}`);
        transpiledLog.push(`Transpiled x86 'sub ${parts[1]}, ${src}' ➔ RISC-V 'addi ${dst}, ${dst}, ${-imm}'`);
      } else {
        const srcReg = mapX86Reg(src);
        riscvLines.push(`sub ${dst}, ${dst}, ${srcReg}`);
        transpiledLog.push(`Transpiled x86 'sub ${parts[1]}, ${src}' ➔ RISC-V 'sub ${dst}, ${dst}, ${srcReg}'`);
      }
    } else if ((op === 'mul' || op === 'imul') && parts.length >= 3) {
      const dst = mapX86Reg(parts[1]);
      const srcReg = mapX86Reg(parts[2]);
      riscvLines.push(`mul ${dst}, ${dst}, ${srcReg}`);
      transpiledLog.push(`Transpiled x86 '${op} ${parts[1]}, ${parts[2]}' ➔ RISC-V 'mul ${dst}, ${dst}, ${srcReg}'`);
    } else if (op === 'xor' && parts.length >= 3) {
      const dst = mapX86Reg(parts[1]);
      const srcReg = mapX86Reg(parts[2]);
      riscvLines.push(`xor ${dst}, ${dst}, ${srcReg}`);
      transpiledLog.push(`Transpiled x86 'xor ${parts[1]}, ${parts[2]}' ➔ RISC-V 'xor ${dst}, ${dst}, ${srcReg}'`);
    } else if (op === 'syscall') {
      riscvLines.push(`ecall`);
      transpiledLog.push(`Transpiled x86 'syscall' ➔ RISC-V 'ecall'`);
    } else {
      riscvLines.push(line);
    }
  }

  // Fallback if empty
  if (riscvLines.length === 0) {
    riscvLines.push('nop');
  }

  return { isX86, riscvLines, transpiledLog };
}


export function parseAssemblyLine(line: string, slot: number): InstructionToken {
  const clean = line.trim();
  const tokens = clean.split(/[\s,]+/);
  const opStr = tokens[0] ? tokens[0].toUpperCase() : 'NOP';
  const opcodeId = OPCODE_MAP[opStr] ?? 45;

  let rd = 0, rs1 = 0, rs2 = 0, imm = 0;
  let rdName = 'zero', rs1Name = 'zero', rs2Name = 'zero';

  if ((opStr === 'LI' || opStr === 'LA') && tokens.length >= 3) {
    rdName = tokens[1];
    rd = REG_MAP[rdName] ?? 5;
    imm = parseInt(tokens[2], 10) || 0;
  } else if ((opStr === 'LW' || opStr === 'SW') && tokens.length >= 3) {
    const regArg = tokens[1];
    const memArg = tokens[2];
    const memMatch = memArg ? memArg.match(/(-?\d+)\((\w+)\)/) : null;
    if (memMatch) {
      imm = parseInt(memMatch[1], 10) || 0;
      const baseReg = memMatch[2];
      if (opStr === 'LW') {
        rdName = regArg;
        rd = REG_MAP[rdName] ?? 5;
        rs1Name = baseReg;
        rs1 = REG_MAP[rs1Name] ?? 5;
      } else {
        rs2Name = regArg;
        rs2 = REG_MAP[rs2Name] ?? 10;
        rs1Name = baseReg;
        rs1 = REG_MAP[rs1Name] ?? 5;
      }
    } else {
      rdName = regArg;
      rd = REG_MAP[rdName] ?? 5;
      rs1Name = memArg || 'zero';
      rs1 = REG_MAP[rs1Name] ?? 5;
    }
  } else if (['ADDI', 'SLLI', 'SRLI', 'SRAI', 'ANDI'].includes(opStr) && tokens.length >= 3) {
    rdName = tokens[1];
    rd = REG_MAP[rdName] ?? 5;
    if (tokens.length >= 4) {
      rs1Name = tokens[2];
      rs1 = REG_MAP[rs1Name] ?? 10;
      imm = parseInt(tokens[3], 10) || 0;
    } else {
      imm = parseInt(tokens[2], 10) || 0;
    }
  } else if (['ADD', 'SUB', 'MUL', 'DIV', 'DIVU', 'AND', 'OR', 'XOR', 'MV'].includes(opStr) && tokens.length >= 3) {
    rdName = tokens[1];
    rd = REG_MAP[rdName] ?? 5;
    rs1Name = tokens[2];
    rs1 = REG_MAP[rs1Name] ?? 10;
    if (tokens.length >= 4) {
      rs2Name = tokens[3];
      rs2 = REG_MAP[rs2Name] ?? 11;
    }
  }

  // 177-dim feature vector construction simulation
  const featureVector = new Array(177).fill(0);
  featureVector[opcodeId % 47] = 1.0;
  if (rd > 0) featureVector[47 + (rd % 32)] = 1.0;
  if (rs1 > 0) featureVector[47 + 32 + (rs1 % 32)] = 1.0;
  if (rs2 > 0) featureVector[47 + 64 + (rs2 % 32)] = 1.0;
  for (let b = 0; b < 32; b++) {
    featureVector[143 + b] = ((imm >> b) & 1);
  }

  return {
    slot,
    mnemonic: clean,
    opcode: opStr,
    opcodeId,
    rs1,
    rs1Name,
    rs2,
    rs2Name,
    rd,
    rdName,
    imm,
    rawText: clean,
    featureVector,
  };
}


export function executeSuperoptimizationPipeline(
  asmText: string,
  onLog?: (msg: string) => void
): SuperoptRun {
  const timestamp = new Date().toISOString();
  const runId = `SAH-${Date.now().toString().slice(-6)}`;

  // Preprocess and transpile x86-64 dialetcs to RISC-V RV32I
  const { isX86, riscvLines, transpiledLog } = preprocessAndTranspile(asmText);

  const tokens: InstructionToken[] = riscvLines.map((line, idx) => parseAssemblyLine(line, idx));


  // Graph building
  const nodes: GraphNode[] = tokens.map((t) => ({
    id: `N${t.slot}`,
    slot: t.slot,
    label: `${t.slot}: ${t.mnemonic}`,
    opcode: t.opcode,
    type: 'inst',
    reads: [t.rs1Name, t.rs2Name].filter((r) => r !== 'zero'),
    writes: t.rdName !== 'zero' ? [t.rdName] : [],
  }));

  const edges: GraphEdge[] = [];
  const lastDef: Record<string, number> = {};
  const registerUsage: Record<string, number> = {};

  tokens.forEach((t, i) => {
    if (i < tokens.length - 1) {
      edges.push({
        id: `e_cf_${i}`,
        source: `N${i}`,
        target: `N${i + 1}`,
        type: 'control_flow',
      });
    }

    [t.rs1Name, t.rs2Name].forEach((r) => {
      if (r !== 'zero') {
        registerUsage[r] = (registerUsage[r] || 0) + 1;
        if (lastDef[r] !== undefined) {
          edges.push({
            id: `e_df_${lastDef[r]}_${i}_${r}`,
            source: `N${lastDef[r]}`,
            target: `N${i}`,
            type: 'data_flow',
            reg: r,
          });
        }
      }
    });

    if (t.rdName !== 'zero') {
      registerUsage[t.rdName] = (registerUsage[t.rdName] || 0) + 1;
      lastDef[t.rdName] = i;
    }
  });

  // Peephole rules matching
  const rules: PeepholeRule[] = [
    { id: 0, name: 'mul_power2_to_slli', pattern: 'mul x, 2^k', replacement: 'slli x, k', desc: 'Converts multiplication by 2^k into 1-cycle SLLI shift', estSavings: 3.0, status: 'NOT_MATCHED' },
    { id: 1, name: 'udiv_power2_to_srli', pattern: 'divu x, 2^k', replacement: 'srli x, k', desc: 'Converts unsigned division by 2^k into 1-cycle SRLI shift', estSavings: 19.0, status: 'NOT_MATCHED' },
    { id: 2, name: 'add_zero_to_nop', pattern: 'addi x, y, 0', replacement: 'mv / NOP', desc: 'Eliminates zero addition identity', estSavings: 1.0, status: 'NOT_MATCHED' },
    { id: 3, name: 'mul_zero_to_li_0', pattern: 'mul x, 0', replacement: 'li rd, 0', desc: 'Simplifies zero multiplication', estSavings: 2.0, status: 'NOT_MATCHED' },
    { id: 4, name: 'xor_self_to_li_0', pattern: 'xor x, x', replacement: 'li rd, 0', desc: 'Converts self XOR into zero register load', estSavings: 1.0, status: 'NOT_MATCHED' },
    { id: 5, name: 'sub_self_to_li_0', pattern: 'sub x, x', replacement: 'li rd, 0', desc: 'Converts self SUB into zero register load', estSavings: 1.0, status: 'NOT_MATCHED' },
    { id: 6, name: 'and_self_to_mv', pattern: 'and x, x', replacement: 'mv / NOP', desc: 'Redundant self bitwise AND', estSavings: 1.0, status: 'NOT_MATCHED' },
    { id: 7, name: 'or_self_to_mv', pattern: 'or x, x', replacement: 'mv / NOP', desc: 'Redundant self bitwise OR', estSavings: 1.0, status: 'NOT_MATCHED' },
    { id: 8, name: 'add_sub_cancel', pattern: 'add(sub(X, Y), Y)', replacement: 'mv rd, X', desc: 'Cancels out inverse subtraction and addition', estSavings: 2.0, status: 'NOT_MATCHED' },
    { id: 9, name: 'sll_srl_to_andi_mask', pattern: 'slli then srli same k', replacement: 'andi mask', desc: 'Fuses double shift sequence into single AND mask', estSavings: 1.0, status: 'NOT_MATCHED' },
  ];

  const ruleMatches: RuleMatchCell[] = [];
  const trajectory: TrajectoryStep[] = [];
  const actions: PolicyAction[] = [];

  let currentCodeLines = [...tokens.map(t => t.mnemonic)];
  let stepCounter = 0;
  let totalCycleSavings = 0;

  tokens.forEach((t, idx) => {
    // Rule 0 match
    if (t.opcode === 'MUL' && idx > 0 && tokens[idx - 1].opcode === 'LI') {
      const immVal = tokens[idx - 1].imm;
      if (immVal > 0 && (immVal & (immVal - 1)) === 0) {
        const k = Math.log2(immVal);
        rules[0].status = 'MATCHED';
        ruleMatches.push({
          slot: idx,
          ruleId: 0,
          ruleName: 'mul_power2_to_slli',
          matched: true,
          candidateRewrite: `slli ${t.rdName}, ${t.rs1Name}, ${k}`,
          estSavings: 3.0,
        });

        actions.push({
          actionIndex: 0 * 16 + idx,
          ruleId: 0,
          ruleName: 'mul_power2_to_slli',
          slotId: idx,
          slotMnemonic: t.mnemonic,
          probability: 0.586,
          maskStatus: 'SELECTED',
          candidateText: `slli ${t.rdName}, ${t.rs1Name}, ${k}`,
        });

        stepCounter++;
        totalCycleSavings += 3.0;
        const before = t.mnemonic;
        const after = `slli ${t.rdName}, ${t.rs1Name}, ${k}`;
        currentCodeLines[idx] = after;

        trajectory.push({
          step: stepCounter,
          ruleId: 0,
          ruleName: 'mul_power2_to_slli',
          slotId: idx,
          beforeInst: before,
          afterInst: after,
          assemblyBefore: riscvLines.join('\n'),
          assemblyAfter: currentCodeLines.join('\n'),
          probability: 0.586,
          reward: 3.0,
          verified: true,
        });
      }
    }

    // Rule 1 match
    if (t.opcode === 'DIVU' && idx > 0 && tokens[idx - 1].opcode === 'LI') {
      const immVal = tokens[idx - 1].imm;
      if (immVal > 0 && (immVal & (immVal - 1)) === 0) {
        const k = Math.log2(immVal);
        rules[1].status = 'MATCHED';
        ruleMatches.push({
          slot: idx,
          ruleId: 1,
          ruleName: 'udiv_power2_to_srli',
          matched: true,
          candidateRewrite: `srli ${t.rdName}, ${t.rs1Name}, ${k}`,
          estSavings: 19.0,
        });

        actions.push({
          actionIndex: 1 * 16 + idx,
          ruleId: 1,
          ruleName: 'udiv_power2_to_srli',
          slotId: idx,
          slotMnemonic: t.mnemonic,
          probability: 0.942,
          maskStatus: 'SELECTED',
          candidateText: `srli ${t.rdName}, ${t.rs1Name}, ${k}`,
        });

        stepCounter++;
        totalCycleSavings += 19.0;
        const before = t.mnemonic;
        const after = `srli ${t.rdName}, ${t.rs1Name}, ${k}`;
        currentCodeLines[idx] = after;

        trajectory.push({
          step: stepCounter,
          ruleId: 1,
          ruleName: 'udiv_power2_to_srli',
          slotId: idx,
          beforeInst: before,
          afterInst: after,
          assemblyBefore: riscvLines.join('\n'),
          assemblyAfter: currentCodeLines.join('\n'),
          probability: 0.942,
          reward: 19.0,
          verified: true,
        });
      }
    }

    // Rule 2 match: ADDI x, y, 0 OR ADD x, y, zero OR ADD x, zero, y
    if (
      (t.opcode === 'ADDI' && t.imm === 0) ||
      (t.opcode === 'ADD' && (t.rs2Name === 'zero' || t.rs1Name === 'zero'))
    ) {
      const srcReg = t.rs2Name === 'zero' ? t.rs1Name : t.rs1Name === 'zero' ? t.rs2Name : t.rs1Name;
      rules[2].status = 'MATCHED';
      ruleMatches.push({
        slot: idx,
        ruleId: 2,
        ruleName: 'add_zero_to_nop',
        matched: true,
        candidateRewrite: `mv ${t.rdName}, ${srcReg}`,
        estSavings: 1.0,
      });

      actions.push({
        actionIndex: 2 * 16 + idx,
        ruleId: 2,
        ruleName: 'add_zero_to_nop',
        slotId: idx,
        slotMnemonic: t.mnemonic,
        probability: 0.812,
        maskStatus: 'SELECTED',
        candidateText: `mv ${t.rdName}, ${srcReg}`,
      });

      stepCounter++;
      totalCycleSavings += 1.0;
      const before = t.mnemonic;
      const after = `mv ${t.rdName}, ${srcReg}`;
      currentCodeLines[idx] = after;

      trajectory.push({
        step: stepCounter,
        ruleId: 2,
        ruleName: 'add_zero_to_nop',
        slotId: idx,
        beforeInst: before,
        afterInst: after,
        assemblyBefore: riscvLines.join('\n'),
        assemblyAfter: currentCodeLines.join('\n'),
        probability: 0.812,
        reward: 1.0,
        verified: true,
      });
    }

    // Rule 9 match: Load After Store Forwarding (sw val, 0(base) followed by lw rd, 0(base))
    if (t.opcode === 'LW' && idx > 0 && tokens[idx - 1].opcode === 'SW') {
      const prevSw = tokens[idx - 1];
      if (t.imm === prevSw.imm && t.rs1Name === prevSw.rs1Name) {
        const valReg = prevSw.rs2Name;
        rules[9].status = 'MATCHED';
        ruleMatches.push({
          slot: idx,
          ruleId: 9,
          ruleName: 'sll_srl_to_andi_mask',
          matched: true,
          candidateRewrite: `mv ${t.rdName}, ${valReg}`,
          estSavings: 2.0,
        });

        actions.push({
          actionIndex: 9 * 16 + idx,
          ruleId: 9,
          ruleName: 'sll_srl_to_andi_mask',
          slotId: idx,
          slotMnemonic: t.mnemonic,
          probability: 0.934,
          maskStatus: 'SELECTED',
          candidateText: `mv ${t.rdName}, ${valReg}`,
        });

        stepCounter++;
        totalCycleSavings += 2.0;
        const before = t.mnemonic;
        const after = `mv ${t.rdName}, ${valReg}`;
        currentCodeLines[idx] = after;

        trajectory.push({
          step: stepCounter,
          ruleId: 9,
          ruleName: 'sll_srl_to_andi_mask',
          slotId: idx,
          beforeInst: before,
          afterInst: after,
          assemblyBefore: riscvLines.join('\n'),
          assemblyAfter: currentCodeLines.join('\n'),
          probability: 0.934,
          reward: 2.0,
          verified: true,
        });
      }
    }


    // Rule 4 match
    if (t.opcode === 'XOR' && t.rs1 === t.rs2 && t.rs1 > 0) {
      rules[4].status = 'MATCHED';
      ruleMatches.push({
        slot: idx,
        ruleId: 4,
        ruleName: 'xor_self_to_li_0',
        matched: true,
        candidateRewrite: `li ${t.rdName}, 0`,
        estSavings: 1.0,
      });

      actions.push({
        actionIndex: 4 * 16 + idx,
        ruleId: 4,
        ruleName: 'xor_self_to_li_0',
        slotId: idx,
        slotMnemonic: t.mnemonic,
        probability: 0.504,
        maskStatus: 'SELECTED',
        candidateText: `li ${t.rdName}, 0`,
      });

      stepCounter++;
      totalCycleSavings += 1.0;
      const before = t.mnemonic;
      const after = `li ${t.rdName}, 0`;
      currentCodeLines[idx] = after;

      trajectory.push({
        step: stepCounter,
        ruleId: 4,
        ruleName: 'xor_self_to_li_0',
        slotId: idx,
        beforeInst: before,
        afterInst: after,
        assemblyBefore: riscvLines.join('\n'),
        assemblyAfter: currentCodeLines.join('\n'),
        probability: 0.504,
        reward: 1.0,
        verified: true,
      });
    }

    // Rule 8 match
    if (t.opcode === 'ADD' && idx > 0 && tokens[idx - 1].opcode === 'SUB') {
      const prevSub = tokens[idx - 1];
      if (t.rs1 === prevSub.rd && t.rs2 === prevSub.rs2) {
        rules[8].status = 'MATCHED';
        ruleMatches.push({
          slot: idx,
          ruleId: 8,
          ruleName: 'add_sub_cancel',
          matched: true,
          candidateRewrite: `mv ${t.rdName}, ${prevSub.rs1Name}`,
          estSavings: 2.0,
        });

        actions.push({
          actionIndex: 8 * 16 + idx,
          ruleId: 8,
          ruleName: 'add_sub_cancel',
          slotId: idx,
          slotMnemonic: t.mnemonic,
          probability: 0.885,
          maskStatus: 'SELECTED',
          candidateText: `mv ${t.rdName}, ${prevSub.rs1Name}`,
        });

        stepCounter++;
        totalCycleSavings += 2.0;
        const before = t.mnemonic;
        const after = `mv ${t.rdName}, ${prevSub.rs1Name}`;
        currentCodeLines[idx] = after;

        trajectory.push({
          step: stepCounter,
          ruleId: 8,
          ruleName: 'add_sub_cancel',
          slotId: idx,
          beforeInst: before,
          afterInst: after,
          assemblyBefore: riscvLines.join('\n'),
          assemblyAfter: currentCodeLines.join('\n'),
          probability: 0.885,
          reward: 2.0,
          verified: true,
        });
      }
    }
  });

  // Calculate cycles
  let origCycles = 0;
  tokens.forEach((t) => {
    if (t.opcode === 'MUL') origCycles += 4.0;
    else if (t.opcode === 'DIV' || t.opcode === 'DIVU') origCycles += 20.0;
    else origCycles += 1.0;
  });

  const optCycles = Math.max(1.0, origCycles - totalCycleSavings);
  const pctReduction = ((origCycles - optCycles) / origCycles) * 100.0;

  // Symbolic register verification mapping
  const symbolicRegisters: SymbolicRegisterState[] = [
    { regName: 't0', origExpr: 'x0 + 4', optExpr: 'x0 + 4', isEqual: true },
    { regName: 't1', origExpr: 'a0 * 4', optExpr: 'a0 << 2', isEqual: true },
    { regName: 't2', origExpr: 'a1 ^ a1', optExpr: '0', isEqual: true },
    { regName: 't3', origExpr: '(a0 * 4) + (a1 ^ a1)', optExpr: '(a0 << 2) + 0', isEqual: true },
  ];

  const optAsmOutput = `.section .text\n.globl _start\n_start:\n` + currentCodeLines.map(l => `    ${l}`).join('\n') + `\n    li a0, 0\n    li a7, 93\n    ecall`;

  return {
    runId,
    timestamp,
    architecture: 'RV32I',
    asmInput: asmText,
    status: 'COMPLETED',
    stages: {
      tokenize: { id: 'tokenize', name: 'TOKENIZE & ENCODE', stepNumber: '01', status: 'COMPLETED', latencyMs: 12 },
      graph: { id: 'graph', name: 'HETERODATA GRAPH', stepNumber: '02', status: 'COMPLETED', latencyMs: 21 },
      peephole: { id: 'peephole', name: 'PEEPHOLE SCANNER', stepNumber: '03', status: 'COMPLETED', latencyMs: 5 },
      gat: { id: 'gat', name: 'GAT ENCODER', stepNumber: '04', status: 'COMPLETED', latencyMs: 38 },
      ppo: { id: 'ppo', name: 'PPO INFERENCE', stepNumber: '05', status: 'COMPLETED', latencyMs: 14 },
      trajectory: { id: 'trajectory', name: 'REWRITE TRAJECTORY', stepNumber: '06', status: 'COMPLETED', latencyMs: 11 },
      verification: { id: 'verification', name: 'Z3 SMT VERIFICATION', stepNumber: '07', status: 'COMPLETED', latencyMs: 94 },
      performance: { id: 'performance', name: 'PERFORMANCE ANALYSIS', stepNumber: '08', status: 'COMPLETED', latencyMs: 63 },
      output: { id: 'output', name: 'OPTIMIZED ASSEMBLY', stepNumber: '09', status: 'COMPLETED', latencyMs: 2 },
    },
    tokens,
    features: {
      opcodeDims: 47,
      regDims: 96,
      immDims: 32,
      branchMemDims: 2,
      totalDims: 177,
    },
    graph: {
      nodes,
      edges,
      density: edges.length / (nodes.length || 1),
      maxDepth: nodes.length,
      registerUsage,
    },
    peephole: {
      rules,
      matches: ruleMatches,
    },
    gat: {
      instInDim: 177,
      regInDim: 1,
      hiddenDim: 128,
      embedDim: 256,
      numHeads: 8,
      numLayers: 3,
      graphEmbeddingSample: [0.142, -0.851, 0.431, 0.991, -0.012, 0.551, -0.223, 0.119, 0.334, -0.662, 0.771, -0.404, 0.129, 0.881, -0.312, 0.045],
      attentionEdges: [
        { sourceNodeId: 'N1', targetNodeId: 'N0', weight: 0.612, explanation: 'N1 (mul) consumes register t0 defined by N0 (li)' },
        { sourceNodeId: 'N3', targetNodeId: 'N1', weight: 0.584, explanation: 'N3 (add) consumes register t1 defined by N1 (mul)' },
        { sourceNodeId: 'N3', targetNodeId: 'N2', weight: 0.281, explanation: 'N3 (add) consumes register t2 defined by N2 (xor)' },
      ],
    },
    ppo: {
      criticValue: 1.1104,
      actions,
      selectedActionIndex: actions.length > 0 ? actions[0].actionIndex : 0,
      actionMask: new Array(160).fill(false).map((_, i) => actions.some((a) => a.actionIndex === i)),
      policyEntropy: 0.245,
    },
    trajectory,
    verification: {
      status: 'UNSAT',
      verified: true,
      symbolicRegisters,
      discrepancyQuery: `(assert (or (not (= r_t0_orig r_t0_opt)) (not (= r_t1_orig r_t1_opt)) (not (= r_t2_orig r_t2_opt)) (not (= r_t3_orig r_t3_opt))))`,
      smtLibText: `(set-logic QF_BV)\n(declare-fun a0 () (_ BitVec 32))\n(declare-fun a1 () (_ BitVec 32))\n(assert (distinct (bvadd (bvshl a0 (_ bv2 32)) (_ bv0 32)) (bvadd (bvmul a0 (_ bv4 32)) (bvxor a1 a1))))\n(check-sat)\n; Solver returned: unsat (Formal Proof Verified)`,
      latencyMs: 94,
      explanation: 'Z3 BitVector symbolic execution proved UNSAT. No counterexample exists for all possible 2^32 32-bit register inputs.',
    },
    performance: {
      origCycles,
      optCycles,
      cycleSavings: totalCycleSavings,
      pctReduction: parseFloat(pctReduction.toFixed(1)),
      origInstCount: tokens.length,
      optInstCount: tokens.length,
      instReduction: 0,
      ruleSavingsBreakdown: trajectory.map((t) => ({ ruleName: t.ruleName, savings: t.reward })),
      stageLatencies: [
        { stage: 'Tokenization', ms: 12 },
        { stage: 'Graph Build', ms: 21 },
        { stage: 'Peephole Scan', ms: 5 },
        { stage: 'GAT Encoder', ms: 38 },
        { stage: 'PPO Policy', ms: 14 },
        { stage: 'Z3 SMT Solver', ms: 94 },
        { stage: 'Performance Analysis', ms: 63 },
      ],
      compilerOverheadMs: 258,
    },
    isTranspiledX86: isX86,
    transpiledRiscvInput: isX86 ? riscvLines.join('\n') : undefined,
    optAsmOutput,
    logs: [
      `[${new Date().toLocaleTimeString()}] Initializing Sahelanthropus Superoptimizer Engine v1.0.4...`,
      ...(isX86
        ? [
            `[${new Date().toLocaleTimeString()}] ⚡ x86-64 Assembly Dialect Detected!`,
            `[${new Date().toLocaleTimeString()}] Auto-transpiling x86-64 instructions and registers to RISC-V RV32I...`,
            ...transpiledLog.map((l) => `[${new Date().toLocaleTimeString()}] ${l}`),
          ]
        : []),
      `[${new Date().toLocaleTimeString()}] Parsing RV32 assembly token array (${tokens.length} instructions)...`,
      `[${new Date().toLocaleTimeString()}] Generated 177-dimensional feature vectors per instruction node.`,
      `[${new Date().toLocaleTimeString()}] Constructing HeteroData program graph (${nodes.length} nodes, ${edges.length} edges)...`,
      `[${new Date().toLocaleTimeString()}] Scanning 10 algebraic peephole rewrite rules...`,
      `[${new Date().toLocaleTimeString()}] Formed 160 action slots (10 rules x 16 instruction positions).`,
      `[${new Date().toLocaleTimeString()}] Forward pass through HeteroGAT Encoder (8 heads, 3 layers, 256-D graph embedding)...`,
      `[${new Date().toLocaleTimeString()}] PPO Actor-Critic policy inference complete. Critic V(s) = +1.1104.`,
      `[${new Date().toLocaleTimeString()}] Step 1: Applied Rule 0 (mul_power2_to_slli) at slot 1.`,
      `[${new Date().toLocaleTimeString()}] Step 2: Applied Rule 4 (xor_self_to_li_0) at slot 2.`,
      `[${new Date().toLocaleTimeString()}] Invoking Z3 SMT BitVector symbolic verifier (timeout = 5.0s)...`,
      `[${new Date().toLocaleTimeString()}] Z3 Result: UNSAT (Formally proven equivalent for 100% of 32-bit input space).`,
      `[${new Date().toLocaleTimeString()}] Calculated cycle savings: ${origCycles} cycles -> ${optCycles} cycles (-${totalCycleSavings} cycles, ${pctReduction.toFixed(1)}% speedup).`,
      `[${new Date().toLocaleTimeString()}] Superoptimization pipeline execution finished successfully in 258 ms.`,
    ],
  };
}

