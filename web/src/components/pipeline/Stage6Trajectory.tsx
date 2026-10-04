// src/components/pipeline/Stage6Trajectory.tsx
import React, { useState } from 'react';
import { GitCommit, ArrowRight, CheckCircle2, Sliders, FileCode, HelpCircle } from 'lucide-react';
import { TrajectoryStep } from '../../types/superoptimizer';

interface Stage6TrajectoryProps {
  trajectory: TrajectoryStep[];
  asmInput: string;
  optAsmOutput: string;
  explainMode?: boolean;
}

export const Stage6Trajectory: React.FC<Stage6TrajectoryProps> = ({
  trajectory,
  asmInput,
  optAsmOutput,
  explainMode = false,
}) => {
  const [activeStep, setActiveStep] = useState<number>(
    trajectory.length > 0 ? trajectory[trajectory.length - 1].step : 0
  );

  const selectedStepObj = trajectory.find((t) => t.step === activeStep);

  return (
    <div className="space-y-4">
      {explainMode && (
        <div className="bg-[#172231] border border-[#36D399]/40 p-3 rounded-xl text-xs text-[#E6EDF3] flex items-start space-x-2">
          <HelpCircle className="w-4 h-4 text-[#36D399] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-[#36D399]">What is the Rewrite Trajectory?</span>
            <p className="text-[#8B9AAF] mt-0.5">
              Superoptimization executes step-by-step rewrites. At each step $t$, a single instruction rewrite is selected, applied, and verified. Drag the timeline slider below to inspect code evolution at any intermediate step.
            </p>
          </div>
        </div>
      )}

      {/* Interactive Step Timeline Slider Bar */}
      <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-[#263445] pb-2">
          <div className="flex items-center space-x-2">
            <GitCommit className="w-4 h-4 text-[#00D9FF]" />
            <h3 className="text-xs font-bold text-[#E6EDF3]">
              TRANSFORMATION TIMELINE SLIDER (t = 0 ... t = {trajectory.length})
            </h3>
          </div>
          <span className="text-[10px] font-mono text-[#00D9FF] bg-[#00D9FF]/10 px-2 py-0.5 rounded border border-[#00D9FF]/30">
            STEP {activeStep} / {trajectory.length}
          </span>
        </div>

        {/* Timeline Slider */}
        <div className="space-y-2 pt-2">
          <input
            type="range"
            min={0}
            max={trajectory.length}
            value={activeStep}
            onChange={(e) => setActiveStep(parseInt(e.target.value, 10))}
            className="w-full accent-[#00D9FF] bg-[#070B12] cursor-pointer"
          />

          <div className="flex justify-between text-xs font-mono text-[#8B9AAF]">
            <span
              onClick={() => setActiveStep(0)}
              className={`cursor-pointer ${activeStep === 0 ? 'text-[#00D9FF] font-bold' : ''}`}
            >
              t0: Original
            </span>
            {trajectory.map((t) => (
              <span
                key={t.step}
                onClick={() => setActiveStep(t.step)}
                className={`cursor-pointer ${activeStep === t.step ? 'text-[#00D9FF] font-bold' : ''}`}
              >
                t{t.step}: {t.ruleName}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Side-by-Side Compiler Diff Viewer */}
      <div className="grid grid-cols-2 gap-4">
        {/* Left Original Assembly */}
        <div className="bg-[#111A26] border border-[#263445] rounded-xl overflow-hidden flex flex-col h-[320px]">
          <div className="bg-[#0D131D] px-4 py-2 border-b border-[#263445] flex justify-between items-center text-xs font-mono">
            <span className="text-[#8B9AAF]">Original Assembly (t = 0)</span>
            <span className="text-[#FF5C6C] font-bold">- Before Rewrites</span>
          </div>
          <pre className="flex-1 bg-[#070B12] p-4 text-xs font-mono text-[#E6EDF3] overflow-y-auto leading-6">
            {asmInput.split('\n').map((line, idx) => (
              <div key={idx} className="hover:bg-[#172231]/40 px-2 rounded">
                <span className="text-[#8B9AAF]/40 select-none mr-4 font-normal">{idx + 1}</span>
                <span>{line}</span>
              </div>
            ))}
          </pre>
        </div>

        {/* Right Current / Optimized Assembly */}
        <div className="bg-[#111A26] border border-[#263445] rounded-xl overflow-hidden flex flex-col h-[320px]">
          <div className="bg-[#0D131D] px-4 py-2 border-b border-[#263445] flex justify-between items-center text-xs font-mono">
            <span className="text-[#8B9AAF]">Transformed Assembly (t = {activeStep})</span>
            <span className="text-[#36D399] font-bold">+ Optimized State</span>
          </div>
          <pre className="flex-1 bg-[#070B12] p-4 text-xs font-mono text-[#E6EDF3] overflow-y-auto leading-6">
            {(selectedStepObj ? selectedStepObj.assemblyAfter : optAsmOutput)
              .split('\n')
              .map((line, idx) => {
                const isModified = selectedStepObj && line.includes(selectedStepObj.afterInst);
                return (
                  <div
                    key={idx}
                    className={`px-2 rounded ${
                      isModified ? 'bg-[#36D399]/20 border-l-2 border-[#36D399]' : 'hover:bg-[#172231]/40'
                    }`}
                  >
                    <span className="text-[#8B9AAF]/40 select-none mr-4 font-normal">{idx + 1}</span>
                    <span className={isModified ? 'text-[#36D399] font-bold' : ''}>{line}</span>
                  </div>
                );
              })}
          </pre>
        </div>
      </div>

      {/* Selected Step Transformation Details */}
      {selectedStepObj && (
        <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-2 text-xs font-mono">
          <div className="flex justify-between items-center border-b border-[#263445] pb-2">
            <span className="text-[#00D9FF] font-bold">
              STEP {selectedStepObj.step}: Rule {selectedStepObj.ruleId} ({selectedStepObj.ruleName}) Applied at Slot {selectedStepObj.slotId}
            </span>
            <span className="text-[#36D399] flex items-center space-x-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              <span>Z3 VERIFIED UNSAT</span>
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 pt-1">
            <div className="bg-[#070B12] p-2.5 rounded border border-[#FF5C6C]/30 text-[#FF5C6C]">
              <span className="text-[10px] text-[#8B9AAF] block">Before Rewrite:</span>
              <span className="font-bold">{selectedStepObj.beforeInst}</span>
            </div>
            <div className="bg-[#070B12] p-2.5 rounded border border-[#36D399]/30 text-[#36D399]">
              <span className="text-[10px] text-[#8B9AAF] block">After Rewrite:</span>
              <span className="font-bold">{selectedStepObj.afterInst}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
