// src/components/performance/Stage8Performance.tsx
import React from 'react';
import { Zap, Clock, TrendingUp, BarChart3, HelpCircle, Layers } from 'lucide-react';
import { PerformanceData } from '../../types/superoptimizer';

interface Stage8PerformanceProps {
  performance: PerformanceData;
  explainMode?: boolean;
}

export const Stage8Performance: React.FC<Stage8PerformanceProps> = ({
  performance,
  explainMode = false,
}) => {
  return (
    <div className="space-y-4">
      {explainMode && (
        <div className="bg-[#172231] border border-[#36D399]/40 p-3 rounded-xl text-xs text-[#E6EDF3] flex items-start space-x-2">
          <HelpCircle className="w-4 h-4 text-[#36D399] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-[#36D399]">Understanding Performance Metrics</span>
            <p className="text-[#8B9AAF] mt-0.5">
              Cycle reductions reflect emulated hardware execution savings on RISC-V pipelines (e.g. replacing 4-cycle multipliers or 20-cycle division logic). We explicitly separate program cycle reduction from compiler search overhead latency.
            </p>
          </div>
        </div>
      )}

      {/* Top 4 Performance Cards */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Original Cycles</span>
          <div className="text-xl font-bold font-mono text-[#FF5C6C]">
            {performance.origCycles.toFixed(1)}
          </div>
          <span className="text-[10px] text-[#8B9AAF]/70">Baseline Execution</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Optimized Cycles</span>
          <div className="text-xl font-bold font-mono text-[#36D399]">
            {performance.optCycles.toFixed(1)}
          </div>
          <span className="text-[10px] text-[#8B9AAF]/70">Superoptimized Execution</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Cycle Savings</span>
          <div className="text-xl font-bold font-mono text-[#00D9FF]">
            -{performance.cycleSavings.toFixed(1)} cycles
          </div>
          <span className="text-[10px] text-[#8B9AAF]/70">{performance.pctReduction}% Faster</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Compiler Overhead</span>
          <div className="text-xl font-bold font-mono text-[#9B7CFF]">
            {performance.compilerOverheadMs} ms
          </div>
          <span className="text-[10px] text-[#8B9AAF]/70">Search & Proof Latency</span>
        </div>
      </div>

      {/* Critical Distinction Banner */}
      <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-3">
          <TrendingUp className="w-5 h-5 text-[#36D399]" />
          <div>
            <span className="font-bold text-[#E6EDF3]">
              PROGRAM PERFORMANCE vs COMPILER OVERHEAD DISTINCTION
            </span>
            <p className="text-[#8B9AAF] text-[11px]">
              The resulting binary executes <strong>{performance.pctReduction}% faster</strong> on hardware. Compiler search and formal proof took <strong>{performance.compilerOverheadMs} ms</strong> offline.
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-4 font-mono text-xs">
          <div className="text-right">
            <span className="text-[#8B9AAF] block">Runtime Gain</span>
            <span className="text-[#36D399] font-bold">-{performance.cycleSavings.toFixed(1)} Cycles</span>
          </div>
          <div className="text-right">
            <span className="text-[#8B9AAF] block">Optimizer Overhead</span>
            <span className="text-[#9B7CFF] font-bold">+{performance.compilerOverheadMs} ms</span>
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-2 gap-4">
        {/* Left Cycle Cost Comparison Bar Chart */}
        <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-[#263445] pb-2">
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-[#00D9FF]" />
              <h3 className="text-xs font-bold text-[#E6EDF3]">EXECUTION CYCLE COST COMPARISON</h3>
            </div>
            <span className="text-[10px] font-mono text-[#36D399]">-{performance.pctReduction}%</span>
          </div>

          <div className="space-y-4 pt-2 font-mono text-xs">
            {/* Original Cycles */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-[#FF5C6C] font-bold">Original Assembly</span>
                <span className="text-[#FF5C6C] font-bold">{performance.origCycles.toFixed(1)} cycles</span>
              </div>
              <div className="w-full bg-[#070B12] h-4 rounded-full overflow-hidden border border-[#263445]">
                <div className="bg-[#FF5C6C] h-full rounded-full w-full" />
              </div>
            </div>

            {/* Optimized Cycles */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-[#36D399] font-bold">Optimized Assembly</span>
                <span className="text-[#36D399] font-bold">{performance.optCycles.toFixed(1)} cycles</span>
              </div>
              <div className="w-full bg-[#070B12] h-4 rounded-full overflow-hidden border border-[#263445]">
                <div
                  className="bg-[#36D399] h-full rounded-full"
                  style={{ width: `${(performance.optCycles / performance.origCycles) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Stage Latency Breakdown Bar Chart */}
        <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-[#263445] pb-2">
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4 text-[#9B7CFF]" />
              <h3 className="text-xs font-bold text-[#E6EDF3]">PIPELINE OVERHEAD STAGE LATENCY</h3>
            </div>
            <span className="text-[10px] font-mono text-[#9B7CFF]">258 MS TOTAL</span>
          </div>

          <div className="space-y-2 font-mono text-xs">
            {performance.stageLatencies.map((st, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-[#8B9AAF]">{st.stage}</span>
                  <span className="text-[#9B7CFF] font-bold">{st.ms} ms</span>
                </div>
                <div className="w-full bg-[#070B12] h-2 rounded-full overflow-hidden border border-[#263445]">
                  <div
                    className="bg-[#9B7CFF] h-full rounded-full"
                    style={{ width: `${(st.ms / 100) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
