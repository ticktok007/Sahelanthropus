// src/types/superoptimizer.ts

export type PipelineStageId =
  | 'tokenize'
  | 'graph'
  | 'peephole'
  | 'gat'
  | 'ppo'
  | 'trajectory'
  | 'verification'
  | 'performance'
  | 'output';

export type StageStatus = 'IDLE' | 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'VERIFICATION_FAILED';

export type UserMode = 'normal' | 'research' | 'explain' | 'expert';

export interface InstructionToken {
  slot: number;
  mnemonic: string;
  opcode: string;
  opcodeId: number;
  rs1: number;
  rs1Name: string;
  rs2: number;
  rs2Name: string;
  rd: number;
  rdName: string;
  imm: number;
  rawText: string;
  featureVector: number[]; // 177-dim
}

export interface FeatureComposition {
  opcodeDims: number; // 47
  regDims: number; // 96 (32x3)
  immDims: number; // 32
  branchMemDims: number; // 2
  totalDims: number; // 177
}

export interface GraphNode {
  id: string;
  slot: number;
  label: string;
  opcode: string;
  type: 'inst' | 'reg';
  reads: string[];
  writes: string[];
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: 'control_flow' | 'data_flow' | 'reads' | 'read_by' | 'writes' | 'written_by';
  reg?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  density: number;
  maxDepth: number;
  registerUsage: Record<string, number>;
}

export interface PeepholeRule {
  id: number;
  name: string;
  pattern: string;
  replacement: string;
  desc: string;
  estSavings: number; // cycles
  status: 'MATCHED' | 'NOT_MATCHED' | 'APPLIED' | 'REJECTED';
}

export interface RuleMatchCell {
  slot: number;
  ruleId: number;
  ruleName: string;
  matched: boolean;
  candidateRewrite: string;
  estSavings: number;
}

export interface AttentionEdge {
  sourceNodeId: string;
  targetNodeId: string;
  weight: number;
  explanation: string;
}

export interface GATState {
  instInDim: number; // 177
  regInDim: number; // 1
  hiddenDim: number; // 128
  embedDim: number; // 256
  numHeads: number; // 8
  numLayers: number; // 3
  graphEmbeddingSample: number[]; // e.g. first 16 values of 256-D tensor
  attentionEdges: AttentionEdge[];
}

export interface PolicyAction {
  actionIndex: number; // 0..159 (rule * 16 + slot)
  ruleId: number;
  ruleName: string;
  slotId: number;
  slotMnemonic: string;
  probability: number;
  maskStatus: 'VALID' | 'MASKED' | 'SELECTED';
  candidateText: string;
}

export interface PPONeuralState {
  criticValue: number;
  actions: PolicyAction[];
  selectedActionIndex: number;
  actionMask: boolean[];
  policyEntropy: number;
}

export interface TrajectoryStep {
  step: number;
  ruleId: number;
  ruleName: string;
  slotId: number;
  beforeInst: string;
  afterInst: string;
  assemblyBefore: string;
  assemblyAfter: string;
  probability: number;
  reward: number;
  verified: boolean;
}

export interface SymbolicRegisterState {
  regName: string;
  origExpr: string;
  optExpr: string;
  isEqual: boolean;
}

export interface VerificationState {
  status: 'UNSAT' | 'SAT' | 'TIMEOUT' | 'UNKNOWN';
  verified: boolean;
  symbolicRegisters: SymbolicRegisterState[];
  discrepancyQuery: string;
  smtLibText: string;
  latencyMs: number;
  explanation: string;
}

export interface PerformanceData {
  origCycles: number;
  optCycles: number;
  cycleSavings: number;
  pctReduction: number;
  origInstCount: number;
  optInstCount: number;
  instReduction: number;
  ruleSavingsBreakdown: { ruleName: string; savings: number }[];
  stageLatencies: { stage: string; ms: number }[];
  compilerOverheadMs: number;
}

export interface StageState {
  id: PipelineStageId;
  name: string;
  stepNumber: string;
  status: StageStatus;
  latencyMs: number;
}

export interface SuperoptRun {
  runId: string;
  timestamp: string;
  architecture: 'RV32I' | 'RV64I';
  asmInput: string;
  status: 'COMPLETED' | 'FAILED' | 'RUNNING' | 'IDLE';
  stages: Record<PipelineStageId, StageState>;
  tokens: InstructionToken[];
  features: FeatureComposition;
  graph: GraphData;
  peephole: {
    rules: PeepholeRule[];
    matches: RuleMatchCell[];
  };
  gat: GATState;
  ppo: PPONeuralState;
  trajectory: TrajectoryStep[];
  verification: VerificationState;
  performance: PerformanceData;
  isTranspiledX86?: boolean;
  transpiledRiscvInput?: string;
  optAsmOutput: string;
  logs: string[];
}


export interface BenchmarkExample {
  id: string;
  name: string;
  description: string;
  code: string;
}
