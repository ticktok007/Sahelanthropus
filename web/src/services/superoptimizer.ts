// src/services/superoptimizer.ts
import type { SuperoptRun, BenchmarkExample } from '../types/superoptimizer';
import { BENCHMARK_EXAMPLES, executeSuperoptimizationPipeline } from './mockService';

// ── Validation ─────────────────────────────────────────────────────────
const VALID_OPCODES = new Set([
  'LI','ADD','ADDI','SUB','MUL','DIV','DIVU','SLLI','SRLI','SRAI',
  'AND','ANDI','OR','ORI','XOR','XORI','MV','NOP','REM','REMU',
  'SLL','SRL','SRA','SLT','SLTI','SLTU','SLTIU','BEQ','BNE','BLT',
  'BGE','BLTU','BGEU','JAL','JALR','J','ECALL','LUI',
  'LB','LH','LW','LD','LBU','LHU','LWU','SB','SH','SW','SD',
]);

const VALID_REGS = new Set([
  'zero','x0','ra','sp','gp','tp',
  't0','t1','t2','t3','t4','t5','t6',
  's0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
  'a0','a1','a2','a3','a4','a5','a6','a7','fp',
  ...Array.from({length:32},(_,i)=>`x${i}`)
]);

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  instructionCount: number;
  registersUsed: string[];
  opcodes: string[];
  immediateCount: number;
}

export function validateAssembly(code: string): ValidationResult {
  const lines = code.split('\n')
    .map(l => l.trim())
    .filter(l => l && !l.startsWith('#') && !l.startsWith('.') && !l.endsWith(':'));

  const errors: string[] = [];
  const regsUsed = new Set<string>();
  const opcodes: string[] = [];
  let immediateCount = 0;

  lines.forEach((line, idx) => {
    const clean = line.includes('#') ? line.split('#')[0].trim() : line;
    if (!clean) return;
    const tokens = clean.split(/[\s,]+/);
    const opStr = tokens[0].toUpperCase();

    if (!VALID_OPCODES.has(opStr)) {
      errors.push(`Line ${idx + 1}: Unknown opcode "${tokens[0]}"`);
      return;
    }
    opcodes.push(opStr);

    // Check register validity
    for (let i = 1; i < tokens.length; i++) {
      const tok = tokens[i].toLowerCase();
      if (VALID_REGS.has(tok)) {
        regsUsed.add(tok);
      } else if (/^-?\d+$/.test(tokens[i]) || /^0x[0-9a-fA-F]+$/.test(tokens[i])) {
        immediateCount++;
      } else if (tok.endsWith(':') || tok === '') {
        // label or empty
      } else if (!VALID_REGS.has(tok)) {
        // Could be a label reference - allow it
      }
    }
  });

  return {
    valid: errors.length === 0 && lines.length > 0,
    errors,
    instructionCount: lines.length,
    registersUsed: Array.from(regsUsed).sort(),
    opcodes,
    immediateCount,
  };
}

// ── Service Interface ──────────────────────────────────────────────────
export interface ISuperoptimizerService {
  runSuperoptimization(
    assembly: string,
    options?: { maxSteps?: number; verify?: boolean },
    onStageUpdate?: (stageId: string, status: string) => void,
    onLog?: (msg: string) => void,
  ): Promise<SuperoptRun>;
  getExamples(): BenchmarkExample[];
}

// ── Mock Adapter ───────────────────────────────────────────────────────
export class MockSuperoptimizerAdapter implements ISuperoptimizerService {
  getExamples(): BenchmarkExample[] {
    return BENCHMARK_EXAMPLES;
  }

  async runSuperoptimization(
    assembly: string,
    _options?: { maxSteps?: number; verify?: boolean },
    onStageUpdate?: (stageId: string, status: string) => void,
    onLog?: (msg: string) => void,
  ): Promise<SuperoptRun> {
    const stageOrder = [
      'tokenize', 'graph', 'peephole', 'gat', 'ppo', 'trajectory', 'verification', 'performance', 'output'
    ];
    const stageTimes = [12, 21, 5, 38, 14, 11, 94, 63, 2];

    // Progressive stage simulation
    for (let i = 0; i < stageOrder.length; i++) {
      onStageUpdate?.(stageOrder[i], 'RUNNING');
      onLog?.(`[${new Date().toLocaleTimeString()}] Stage ${String(i + 1).padStart(2, '0')}: ${stageOrder[i].toUpperCase()} running...`);
      await new Promise(r => setTimeout(r, stageTimes[i] + Math.random() * 50));
      onStageUpdate?.(stageOrder[i], 'COMPLETED');
    }

    // Execute the mock pipeline
    const result = executeSuperoptimizationPipeline(assembly, onLog);
    return result;
  }
}

// ── Real Backend Adapter (Stub) ────────────────────────────────────────
export class RealSuperoptimizerAdapter implements ISuperoptimizerService {
  private baseUrl: string;

  constructor(baseUrl: string = '/api') {
    this.baseUrl = baseUrl;
  }

  getExamples(): BenchmarkExample[] {
    return BENCHMARK_EXAMPLES;
  }

  async runSuperoptimization(
    assembly: string,
    options?: { maxSteps?: number; verify?: boolean },
    onStageUpdate?: (stageId: string, status: string) => void,
    onLog?: (msg: string) => void,
  ): Promise<SuperoptRun> {
    onLog?.(`[${new Date().toLocaleTimeString()}] Connecting to backend at ${this.baseUrl}...`);
    try {
      const response = await fetch(`${this.baseUrl}/superoptimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          architecture: 'riscv32',
          assembly,
          options: { max_steps: options?.maxSteps ?? 10, verify: options?.verify ?? true, performance_analysis: true },
        }),
      });
      if (!response.ok) throw new Error(`Backend returned ${response.status}`);
      return await response.json();
    } catch (err) {
      onLog?.(`[${new Date().toLocaleTimeString()}] Backend unavailable: ${err}. Falling back to mock.`);
      const mock = new MockSuperoptimizerAdapter();
      return mock.runSuperoptimization(assembly, options, onStageUpdate, onLog);
    }
  }
}

// ── Factory ────────────────────────────────────────────────────────────
export function createService(mode: 'mock' | 'real' = 'mock'): ISuperoptimizerService {
  if (mode === 'real') return new RealSuperoptimizerAdapter();
  return new MockSuperoptimizerAdapter();
}
