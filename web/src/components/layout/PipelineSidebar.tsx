// src/components/layout/PipelineSidebar.tsx
import React from 'react';
import {
  Code2,
  Network,
  Binary,
  Layers,
  BrainCircuit,
  GitCommit,
  ShieldCheck,
  Zap,
  FileCode,
  CheckCircle2,
  Loader2,
  AlertCircle
} from 'lucide-react';
import { PipelineStageId, StageState } from '../../types/superoptimizer';

interface PipelineSidebarProps {
  stages: Record<PipelineStageId, StageState>;
  activeStage: PipelineStageId;
  onSelectStage: (stage: PipelineStageId) => void;
}

const STAGE_ICONS: Record<PipelineStageId, React.ReactNode> = {
  tokenize: <Code2 className="w-4 h-4" />,
  graph: <Network className="w-4 h-4" />,
  peephole: <Binary className="w-4 h-4" />,
  gat: <Layers className="w-4 h-4" />,
  ppo: <BrainCircuit className="w-4 h-4" />,
  trajectory: <GitCommit className="w-4 h-4" />,
  verification: <ShieldCheck className="w-4 h-4" />,
  performance: <Zap className="w-4 h-4" />,
  output: <FileCode className="w-4 h-4" />,
};

export const PipelineSidebar: React.FC<PipelineSidebarProps> = ({
  stages,
  activeStage,
  onSelectStage,
}) => {
  const stageKeys: PipelineStageId[] = [
    'tokenize',
    'graph',
    'peephole',
    'gat',
    'ppo',
    'trajectory',
    'verification',
    'performance',
    'output',
  ];

  return (
    <aside className="w-60 bg-[#0D131D] border-r border-[#263445] flex flex-col justify-between select-none">
      <div className="p-3">
        <div className="flex items-center justify-between px-2 pb-3 mb-2 border-b border-[#263445]">
          <span className="text-[11px] font-semibold text-[#8B9AAF] tracking-wider uppercase">
            PIPELINE STAGES
          </span>
          <span className="text-[10px] font-mono text-[#00D9FF] bg-[#00D9FF]/10 px-1.5 py-0.5 rounded border border-[#00D9FF]/30">
            8 / 8 READY
          </span>
        </div>

        <nav className="space-y-1">
          {stageKeys.map((key) => {
            const s = stages[key];
            const isActive = activeStage === key;

            return (
              <button
                key={key}
                onClick={() => onSelectStage(key)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-all flex items-center justify-between group ${
                  isActive
                    ? 'bg-[#172231] border border-[#00D9FF]/40 text-[#E6EDF3] shadow-md shadow-[#00D9FF]/5'
                    : 'hover:bg-[#111A26] border border-transparent text-[#8B9AAF] hover:text-[#E6EDF3]'
                }`}
              >
                <div className="flex items-center space-x-2.5">
                  <span
                    className={`font-mono text-[10px] w-5 text-center ${
                      isActive ? 'text-[#00D9FF]' : 'text-[#8B9AAF]'
                    }`}
                  >
                    {s.stepNumber}
                  </span>
                  <div
                    className={`${
                      isActive ? 'text-[#00D9FF]' : 'text-[#8B9AAF] group-hover:text-[#E6EDF3]'
                    }`}
                  >
                    {STAGE_ICONS[key]}
                  </div>
                  <span className="truncate max-w-[100px]">{s.name}</span>
                </div>

                <div className="flex items-center space-x-1.5 font-mono text-[10px]">
                  {s.status === 'COMPLETED' && (
                    <>
                      <span className="text-[#36D399] font-normal">{s.latencyMs}ms</span>
                      <CheckCircle2 className="w-3.5 h-3.5 text-[#36D399]" />
                    </>
                  )}
                  {s.status === 'RUNNING' && (
                    <Loader2 className="w-3.5 h-3.5 text-[#00D9FF] animate-spin" />
                  )}
                  {s.status === 'FAILED' && (
                    <AlertCircle className="w-3.5 h-3.5 text-[#FF5C6C]" />
                  )}
                  {s.status === 'IDLE' && (
                    <span className="text-[#263445]">--</span>
                  )}
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* System Hardware Status Widget */}
      <div className="p-3 m-3 bg-[#111A26] border border-[#263445] rounded-lg text-[11px] font-mono space-y-2">
        <div className="flex justify-between text-[#8B9AAF]">
          <span>GPU Device:</span>
          <span className="text-[#E6EDF3]">NVIDIA RTX</span>
        </div>
        <div className="flex justify-between text-[#8B9AAF]">
          <span>Formal Engine:</span>
          <span className="text-[#36D399]">Z3 SMT v4.12</span>
        </div>
        <div className="flex justify-between text-[#8B9AAF]">
          <span>ISA Target:</span>
          <span className="text-[#00D9FF]">RISC-V RV32I</span>
        </div>
      </div>
    </aside>
  );
};
