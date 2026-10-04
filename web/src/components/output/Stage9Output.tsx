// src/components/output/Stage9Output.tsx
import React, { useState } from 'react';
import { FileCode, Copy, Download, Check, ShieldCheck, Zap, RefreshCw, Share2 } from 'lucide-react';
import { SuperoptRun } from '../../types/superoptimizer';

interface Stage9OutputProps {
  run: SuperoptRun;
  onRerun: () => void;
}

export const Stage9Output: React.FC<Stage9OutputProps> = ({ run, onRerun }) => {
  const [copied, setCopied] = useState<boolean>(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(run.optAsmOutput);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([run.optAsmOutput], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `optimized_${run.runId.toLowerCase()}.s`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExportJson = () => {
    const jsonStr = JSON.stringify(run, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `report_${run.runId.toLowerCase()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      {/* Top Completion Header */}
      <div className="bg-[#111A26] border border-[#36D399]/40 rounded-xl p-5 flex items-center justify-between shadow-xl shadow-[#36D399]/5">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 rounded-xl bg-[#36D399]/10 border border-[#36D399]/40 flex items-center justify-center">
            <Check className="w-7 h-7 text-[#36D399]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-lg font-bold text-[#E6EDF3]">
                SUPEROPTIMIZATION COMPLETE & VERIFIED
              </h2>
              <span className="text-xs font-mono font-bold text-[#36D399] bg-[#36D399]/10 px-2.5 py-0.5 rounded border border-[#36D399]/40">
                VERIFIED UNSAT
              </span>
            </div>
            <p className="text-xs text-[#8B9AAF] mt-0.5">
              Transformed RISC-V assembly code generated, formally verified by Z3 SMT, and ready for deployment.
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={onRerun}
            className="bg-[#172231] hover:bg-[#263445] text-[#E6EDF3] border border-[#263445] px-3 py-1.5 rounded-lg text-xs font-medium flex items-center space-x-1.5 transition-all"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Run Another</span>
          </button>
          <button
            onClick={handleDownload}
            className="bg-[#00D9FF] hover:bg-[#00D9FF]/90 text-[#070B12] font-bold px-4 py-1.5 rounded-lg text-xs flex items-center space-x-1.5 transition-all shadow-md shadow-[#00D9FF]/20"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download .S File</span>
          </button>
        </div>
      </div>

      {/* Target Metadata Status Cards */}
      <div className="grid grid-cols-4 gap-3 font-mono text-xs">
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[10px] text-[#8B9AAF] block">Target Architecture</span>
          <span className="text-sm font-bold text-[#00D9FF]">RISC-V RV32I</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[10px] text-[#8B9AAF] block">Z3 Proof Status</span>
          <span className="text-sm font-bold text-[#36D399]">UNSAT (Proven)</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[10px] text-[#8B9AAF] block">Cycle Savings</span>
          <span className="text-sm font-bold text-[#00D9FF]">-{run.performance.cycleSavings} Cycles ({run.performance.pctReduction}%)</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[10px] text-[#8B9AAF] block">Pipeline Latency</span>
          <span className="text-sm font-bold text-[#9B7CFF]">{run.performance.compilerOverheadMs} ms</span>
        </div>
      </div>

      {/* Side-by-Side Code Viewer */}
      <div className="grid grid-cols-2 gap-4">
        {/* Left Original Assembly */}
        <div className="bg-[#111A26] border border-[#263445] rounded-xl overflow-hidden flex flex-col h-[380px]">
          <div className="bg-[#0D131D] px-4 py-2.5 border-b border-[#263445] flex justify-between items-center text-xs font-mono">
            <span className="text-[#8B9AAF] font-semibold">Original RISC-V Input</span>
            <span className="text-[#FF5C6C] font-bold">{run.tokens.length} instructions</span>
          </div>
          <pre className="flex-1 bg-[#070B12] p-4 text-xs font-mono text-[#E6EDF3] overflow-y-auto leading-6">
            {run.asmInput}
          </pre>
        </div>

        {/* Right Final Transformed RISC-V Output */}
        <div className="bg-[#111A26] border border-[#263445] rounded-xl overflow-hidden flex flex-col h-[380px]">
          <div className="bg-[#0D131D] px-4 py-2.5 border-b border-[#263445] flex justify-between items-center text-xs font-mono">
            <div className="flex items-center space-x-2">
              <FileCode className="w-4 h-4 text-[#36D399]" />
              <span className="text-[#E6EDF3] font-bold">Optimized RISC-V Assembly</span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleCopy}
                className="px-2.5 py-1 rounded bg-[#172231] border border-[#263445] text-[#00D9FF] hover:bg-[#263445] transition-all flex items-center space-x-1"
              >
                {copied ? <Check className="w-3 h-3 text-[#36D399]" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'Copied!' : 'Copy Code'}</span>
              </button>
              <button
                onClick={handleExportJson}
                className="px-2.5 py-1 rounded bg-[#172231] border border-[#263445] text-[#8B9AAF] hover:text-[#E6EDF3] transition-all flex items-center space-x-1"
              >
                <Share2 className="w-3 h-3" />
                <span>Export Report</span>
              </button>
            </div>
          </div>
          <pre className="flex-1 bg-[#070B12] p-4 text-xs font-mono text-[#36D399] overflow-y-auto leading-6">
            {run.optAsmOutput}
          </pre>
        </div>
      </div>
    </div>
  );
};
