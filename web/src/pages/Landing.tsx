// src/pages/Landing.tsx
import React from 'react';
import { Cpu, Play, ShieldCheck, Zap, Brain, ArrowRight, Code2, Network, CheckCircle2, Layers } from 'lucide-react';
import { BenchmarkExample } from '../types/superoptimizer';
import { BENCHMARK_EXAMPLES } from '../services/mockService';

interface LandingProps {
  onOpenWorkbench: () => void;
  onSelectExample: (ex: BenchmarkExample) => void;
}

export const Landing: React.FC<LandingProps> = ({ onOpenWorkbench, onSelectExample }) => {
  return (
    <div className="min-h-screen bg-[#070B12] text-[#E6EDF3] flex flex-col justify-between selection:bg-[#00D9FF]/30">
      {/* Landing Top Header */}
      <header className="h-16 border-b border-[#263445] px-8 flex items-center justify-between bg-[#0D131D]">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#00D9FF] to-[#9B7CFF] p-0.5 flex items-center justify-center shadow-lg shadow-[#00D9FF]/20">
            <div className="w-full h-full bg-[#070B12] rounded-[10px] flex items-center justify-center">
              <Cpu className="w-5 h-5 text-[#00D9FF]" />
            </div>
          </div>
          <div>
            <h1 className="text-base font-bold tracking-wider text-[#E6EDF3] uppercase flex items-center gap-2">
              SAHELANTHROPUS
              <span className="text-[10px] bg-[#172231] border border-[#263445] text-[#00D9FF] px-2 py-0.5 rounded font-mono font-normal">
                v1.0.4
              </span>
            </h1>
            <p className="text-[11px] text-[#8B9AAF]">Interactive RISC-V Neural Superoptimizer & Formal Verification</p>
          </div>
        </div>

        <button
          onClick={onOpenWorkbench}
          className="bg-[#00D9FF] hover:bg-[#00D9FF]/90 text-[#070B12] font-bold px-5 py-2 rounded-xl text-xs flex items-center space-x-2 transition-all shadow-lg shadow-[#00D9FF]/20 active:scale-95"
        >
          <Play className="w-4 h-4 fill-current" />
          <span>OPEN SUPEROPTIMIZER WORKBENCH</span>
        </button>
      </header>

      {/* Hero Section */}
      <main className="max-w-6xl mx-auto px-6 py-12 flex-1 flex flex-col justify-center space-y-12">
        <div className="text-center space-y-4 max-w-3xl mx-auto">
          <div className="inline-flex items-center space-x-2 bg-[#172231] border border-[#00D9FF]/30 text-[#00D9FF] px-3.5 py-1 rounded-full text-xs font-mono">
            <ShieldCheck className="w-3.5 h-3.5 text-[#36D399]" />
            <span>Formally Verified RISC-V Assembly Superoptimization</span>
          </div>

          <h1 className="text-4xl font-extrabold tracking-tight text-[#E6EDF3] sm:text-5xl leading-tight">
            Discover Faster Instruction Sequences.<br />
            <span className="bg-gradient-to-r from-[#00D9FF] via-[#4D8DFF] to-[#9B7CFF] bg-clip-text text-transparent">
              Verify Every Transformation.
            </span>
          </h1>

          <p className="text-sm text-[#8B9AAF] max-w-2xl mx-auto leading-relaxed">
            Sahelanthropus integrates Heterogeneous Graph Neural Networks (HeteroGAT), Proximal Policy Optimization (PPO), algebraic peephole rulebooks, and Z3 BitVector SMT symbolic execution to superoptimize RISC-V assembly with 100% formal equivalence proofs.
          </p>

          <div className="pt-4 flex items-center justify-center space-x-4">
            <button
              onClick={onOpenWorkbench}
              className="bg-[#00D9FF] hover:bg-[#00D9FF]/90 text-[#070B12] font-extrabold px-6 py-3 rounded-xl text-sm flex items-center space-x-2 transition-all shadow-xl shadow-[#00D9FF]/25 active:scale-95"
            >
              <span>ENTER WORKSPACE</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => {
                onSelectExample(BENCHMARK_EXAMPLES[4]);
                onOpenWorkbench();
              }}
              className="bg-[#111A26] border border-[#263445] hover:border-[#00D9FF]/50 text-[#E6EDF3] px-6 py-3 rounded-xl text-sm font-semibold flex items-center space-x-2 transition-all"
            >
              <Zap className="w-4 h-4 text-[#FFB84D]" />
              <span>LOAD FEATURED DEMO (EXAMPLE 5)</span>
            </button>
          </div>
        </div>

        {/* 8-Stage Miniature Pipeline Flow Diagram */}
        <div className="bg-[#0D131D] border border-[#263445] rounded-2xl p-6 space-y-4 shadow-2xl">
          <div className="flex justify-between items-center border-b border-[#263445] pb-3">
            <span className="text-xs font-bold text-[#E6EDF3] tracking-wide uppercase font-mono">
              THE 8-STAGE SUPEROPTIMIZATION PIPELINE
            </span>
            <span className="text-[11px] font-mono text-[#36D399]">100% TRANSPARENT COMPILER ENGINE</span>
          </div>

          <div className="grid grid-cols-8 gap-2 font-mono text-center text-xs">
            <div className="bg-[#111A26] p-3 rounded-xl border border-[#263445] space-y-1">
              <span className="text-[10px] text-[#00D9FF]">01</span>
              <span className="text-[#E6EDF3] font-bold block text-[11px]">Tokenize</span>
              <span className="text-[9px] text-[#8B9AAF] block">177-D Vector</span>
            </div>

            <div className="bg-[#111A26] p-3 rounded-xl border border-[#263445] space-y-1">
              <span className="text-[10px] text-[#00D9FF]">02</span>
              <span className="text-[#E6EDF3] font-bold block text-[11px]">Graph</span>
              <span className="text-[9px] text-[#8B9AAF] block">Def-Use Edges</span>
            </div>

            <div className="bg-[#111A26] p-3 rounded-xl border border-[#263445] space-y-1">
              <span className="text-[10px] text-[#00D9FF]">03</span>
              <span className="text-[#E6EDF3] font-bold block text-[11px]">Peephole</span>
              <span className="text-[9px] text-[#8B9AAF] block">10 Rules</span>
            </div>

            <div className="bg-[#111A26] p-3 rounded-xl border border-[#263445] space-y-1">
              <span className="text-[10px] text-[#9B7CFF]">04</span>
              <span className="text-[#E6EDF3] font-bold block text-[11px]">GAT</span>
              <span className="text-[9px] text-[#8B9AAF] block">8 Attention Heads</span>
            </div>

            <div className="bg-[#111A26] p-3 rounded-xl border border-[#263445] space-y-1">
              <span className="text-[10px] text-[#9B7CFF]">05</span>
              <span className="text-[#E6EDF3] font-bold block text-[11px]">PPO Policy</span>
              <span className="text-[9px] text-[#8B9AAF] block">Action Logits</span>
            </div>

            <div className="bg-[#111A26] p-3 rounded-xl border border-[#263445] space-y-1">
              <span className="text-[10px] text-[#00D9FF]">06</span>
              <span className="text-[#E6EDF3] font-bold block text-[11px]">Rewrite</span>
              <span className="text-[9px] text-[#8B9AAF] block">Trajectory Step</span>
            </div>

            <div className="bg-[#111A26] p-3 rounded-xl border border-[#36D399]/40 space-y-1">
              <span className="text-[10px] text-[#36D399]">07</span>
              <span className="text-[#36D399] font-bold block text-[11px]">Z3 SMT</span>
              <span className="text-[9px] text-[#36D399] block">UNSAT Proof</span>
            </div>

            <div className="bg-[#111A26] p-3 rounded-xl border border-[#263445] space-y-1">
              <span className="text-[10px] text-[#36D399]">08</span>
              <span className="text-[#E6EDF3] font-bold block text-[11px]">Performance</span>
              <span className="text-[9px] text-[#8B9AAF] block">Cycle Cost</span>
            </div>
          </div>
        </div>

        {/* Informational Core Metric Cards */}
        <div className="grid grid-cols-6 gap-3 font-mono text-xs">
          <div className="bg-[#111A26] border border-[#263445] p-3.5 rounded-xl space-y-1">
            <span className="text-[#8B9AAF] text-[10px] block">FEATURE SPACE</span>
            <span className="text-base font-bold text-[#00D9FF]">177-D</span>
            <span className="text-[9px] text-[#8B9AAF]/60 block">Opcode + Regs + Imm</span>
          </div>

          <div className="bg-[#111A26] border border-[#263445] p-3.5 rounded-xl space-y-1">
            <span className="text-[#8B9AAF] text-[10px] block">RULEBOOK</span>
            <span className="text-base font-bold text-[#FFB84D]">10 RULES</span>
            <span className="text-[9px] text-[#8B9AAF]/60 block">Algebraic Rewrites</span>
          </div>

          <div className="bg-[#111A26] border border-[#263445] p-3.5 rounded-xl space-y-1">
            <span className="text-[#8B9AAF] text-[10px] block">BACKBONE</span>
            <span className="text-base font-bold text-[#9B7CFF]">HeteroGAT</span>
            <span className="text-[9px] text-[#8B9AAF]/60 block">3-Layer GraphConv</span>
          </div>

          <div className="bg-[#111A26] border border-[#263445] p-3.5 rounded-xl space-y-1">
            <span className="text-[#8B9AAF] text-[10px] block">POLICY</span>
            <span className="text-base font-bold text-[#9B7CFF]">PPO RL</span>
            <span className="text-[9px] text-[#8B9AAF]/60 block">GAE-lambda Advantage</span>
          </div>

          <div className="bg-[#111A26] border border-[#263445] p-3.5 rounded-xl space-y-1">
            <span className="text-[#8B9AAF] text-[10px] block">VERIFIER</span>
            <span className="text-base font-bold text-[#36D399]">Z3 SMT</span>
            <span className="text-[9px] text-[#8B9AAF]/60 block">BitVector Proof</span>
          </div>

          <div className="bg-[#111A26] border border-[#263445] p-3.5 rounded-xl space-y-1">
            <span className="text-[#8B9AAF] text-[10px] block">TARGET ISA</span>
            <span className="text-base font-bold text-[#00D9FF]">RISC-V RV32</span>
            <span className="text-[9px] text-[#8B9AAF]/60 block">32-Bit Base Architecture</span>
          </div>
        </div>

        {/* Preset Benchmark Examples Cards */}
        <div className="space-y-3">
          <span className="text-xs font-bold text-[#E6EDF3] tracking-wide uppercase font-mono block">
            PRE-PACKAGED BENCHMARK EXAMPLES
          </span>

          <div className="grid grid-cols-3 gap-3 font-mono text-xs">
            {BENCHMARK_EXAMPLES.slice(0, 6).map((ex) => (
              <div
                key={ex.id}
                onClick={() => {
                  onSelectExample(ex);
                  onOpenWorkbench();
                }}
                className="bg-[#111A26] border border-[#263445] hover:border-[#00D9FF] p-4 rounded-xl space-y-2 cursor-pointer transition-all group"
              >
                <div className="flex justify-between items-center">
                  <span className="text-[#00D9FF] font-bold text-xs group-hover:underline">
                    {ex.name}
                  </span>
                  <ArrowRight className="w-3.5 h-3.5 text-[#8B9AAF] group-hover:text-[#00D9FF] transition-colors" />
                </div>
                <p className="text-[11px] text-[#8B9AAF] font-sans leading-normal">
                  {ex.description}
                </p>
                <pre className="bg-[#070B12] p-2 rounded text-[10px] text-[#36D399] leading-4 border border-[#263445]">
                  {ex.code}
                </pre>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="h-12 border-t border-[#263445] px-8 flex items-center justify-between bg-[#0D131D] text-[11px] text-[#8B9AAF] font-mono">
        <span>Sahelanthropus Superoptimizer Engine v1.0.4</span>
        <span>Google DeepMind Research Architecture</span>
      </footer>
    </div>
  );
};
