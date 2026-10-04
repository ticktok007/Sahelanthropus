// src/components/pipeline/PipelineFlowViewer.tsx
import React from 'react';
import { ChevronRight, CheckCircle2 } from 'lucide-react';
import { PipelineStageId, StageState } from '../../types/superoptimizer';

interface PipelineFlowViewerProps {
  stages: Record<PipelineStageId, StageState>;
  activeStage: PipelineStageId;
  onSelectStage: (stage: PipelineStageId) => void;
}

export const PipelineFlowViewer: React.FC<PipelineFlowViewerProps> = ({
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
    <div className="bg-[#0D131D] border-b border-[#263445] px-4 py-2.5 overflow-x-auto select-none">
      <div className="flex items-center space-x-2 min-w-max">
        {stageKeys.map((key, idx) => {
          const s = stages[key];
          const isActive = activeStage === key;

          return (
            <React.Fragment key={key}>
              <div
                onClick={() => onSelectStage(key)}
                className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg border text-xs font-mono cursor-pointer transition-all ${
                  isActive
                    ? 'bg-[#172231] border-[#00D9FF] text-[#00D9FF] font-bold shadow-md shadow-[#00D9FF]/10'
                    : 'bg-[#111A26] border-[#263445] text-[#8B9AAF] hover:border-[#8B9AAF]/50 hover:text-[#E6EDF3]'
                }`}
              >
                <span className="text-[10px] text-[#8B9AAF]">{s.stepNumber}</span>
                <span className="truncate">{s.name}</span>
                {s.status === 'COMPLETED' && (
                  <CheckCircle2 className="w-3.5 h-3.5 text-[#36D399]" />
                )}
              </div>

              {idx < stageKeys.length - 1 && (
                <ChevronRight className="w-4 h-4 text-[#263445] shrink-0" />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
