// src/pages/Workbench.tsx
import React, { useState } from 'react';
import { AppHeader } from '../components/layout/AppHeader';
import { PipelineSidebar } from '../components/layout/PipelineSidebar';
import { PipelineFlowViewer } from '../components/pipeline/PipelineFlowViewer';
import { AssemblyEditor } from '../components/assembly/AssemblyEditor';
import { InputValidationPanel } from '../components/assembly/InputValidationPanel';
import { Stage1Tokenization } from '../components/analysis/Stage1Tokenization';
import { Stage2ProgramGraph } from '../components/graph/Stage2ProgramGraph';
import { Stage3Peephole } from '../components/pipeline/Stage3Peephole';
import { Stage4GAT } from '../components/neural/Stage4GAT';
import { Stage5PPO } from '../components/neural/Stage5PPO';
import { Stage6Trajectory } from '../components/pipeline/Stage6Trajectory';
import { Stage7Verification } from '../components/verification/Stage7Verification';
import { Stage8Performance } from '../components/performance/Stage8Performance';
import { Stage9Output } from '../components/output/Stage9Output';
import { LogConsole } from '../components/layout/LogConsole';
import { RunHistoryView } from '../components/runs/RunHistoryView';

import { PipelineStageId, UserMode, SuperoptRun, BenchmarkExample } from '../types/superoptimizer';
import { BENCHMARK_EXAMPLES, executeSuperoptimizationPipeline } from '../services/mockService';

interface WorkbenchProps {
  initialCode?: string;
  onGoToLanding?: () => void;
}

export const Workbench: React.FC<WorkbenchProps> = ({
  initialCode = BENCHMARK_EXAMPLES[4].code,
  onGoToLanding,
}) => {
  const [asmCode, setAsmCode] = useState<string>(initialCode);
  const [activeStage, setActiveStage] = useState<PipelineStageId>('tokenize');
  const [userMode, setUserMode] = useState<UserMode>('normal');
  const [isHistoryOpen, setIsHistoryOpen] = useState<boolean>(false);

  // Maintain run trajectory history
  const [currentRun, setCurrentRun] = useState<SuperoptRun>(() =>
    executeSuperoptimizationPipeline(initialCode)
  );
  const [historyRuns, setHistoryRuns] = useState<SuperoptRun[]>([currentRun]);

  const handleRunSuperopt = () => {
    const newRun = executeSuperoptimizationPipeline(asmCode);
    setCurrentRun(newRun);
    setHistoryRuns((prev) => [newRun, ...prev]);
  };

  const handleSelectExample = (ex: BenchmarkExample) => {
    setAsmCode(ex.code);
    const newRun = executeSuperoptimizationPipeline(ex.code);
    setCurrentRun(newRun);
    setHistoryRuns((prev) => [newRun, ...prev]);
  };

  return (
    <div className="flex flex-col h-screen bg-[#070B12] text-[#E6EDF3] overflow-hidden select-none font-sans">
      {/* Top Application Header */}
      <AppHeader
        runId={currentRun.runId}
        architecture={currentRun.architecture}
        status={currentRun.status}
        isVerified={currentRun.verification.verified}
        userMode={userMode}
        onModeChange={setUserMode}
        onRunSuperopt={handleRunSuperopt}
        onToggleHistory={() => setIsHistoryOpen(!isHistoryOpen)}
        isHistoryOpen={isHistoryOpen}
        onGoToLanding={onGoToLanding}
      />


      {/* Main Workbench Layout: Left Sidebar + Central Stage Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Fixed Pipeline Sidebar */}
        <PipelineSidebar
          stages={currentRun.stages}
          activeStage={activeStage}
          onSelectStage={setActiveStage}
        />

        {/* Central Workspace Area */}
        <div className="flex-1 flex flex-col overflow-hidden bg-[#070B12]">
          {/* Top Stage Pipeline Flow Stepper */}
          <PipelineFlowViewer
            stages={currentRun.stages}
            activeStage={activeStage}
            onSelectStage={setActiveStage}
          />

          {/* Main Stage Content Container */}
          <div className="flex-1 p-5 overflow-y-auto space-y-4">
            {/* Input Workspace View */}
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="col-span-2 h-[260px]">
                <AssemblyEditor
                  code={asmCode}
                  onChange={setAsmCode}
                  onRun={handleRunSuperopt}
                  onSelectExample={handleSelectExample}
                />
              </div>
              <div className="h-[260px]">
                <InputValidationPanel
                  tokens={currentRun.tokens}
                  isTranspiledX86={currentRun.isTranspiledX86}
                  transpiledRiscvInput={currentRun.transpiledRiscvInput}
                />

              </div>
            </div>

            {/* Stage View Conditionals */}
            {activeStage === 'tokenize' && (
              <Stage1Tokenization
                tokens={currentRun.tokens}
                features={currentRun.features}
                explainMode={userMode === 'explain'}
              />
            )}

            {activeStage === 'graph' && (
              <Stage2ProgramGraph
                graph={currentRun.graph}
                explainMode={userMode === 'explain'}
              />
            )}

            {activeStage === 'peephole' && (
              <Stage3Peephole
                rules={currentRun.peephole.rules}
                matches={currentRun.peephole.matches}
                tokens={currentRun.tokens}
                explainMode={userMode === 'explain'}
              />
            )}

            {activeStage === 'gat' && (
              <Stage4GAT
                gat={currentRun.gat}
                explainMode={userMode === 'explain'}
              />
            )}

            {activeStage === 'ppo' && (
              <Stage5PPO
                ppo={currentRun.ppo}
                explainMode={userMode === 'explain'}
              />
            )}

            {activeStage === 'trajectory' && (
              <Stage6Trajectory
                trajectory={currentRun.trajectory}
                asmInput={currentRun.asmInput}
                optAsmOutput={currentRun.optAsmOutput}
                explainMode={userMode === 'explain'}
              />
            )}

            {activeStage === 'verification' && (
              <Stage7Verification
                verification={currentRun.verification}
                explainMode={userMode === 'explain'}
              />
            )}

            {activeStage === 'performance' && (
              <Stage8Performance
                performance={currentRun.performance}
                explainMode={userMode === 'explain'}
              />
            )}

            {activeStage === 'output' && (
              <Stage9Output
                run={currentRun}
                onRerun={handleRunSuperopt}
              />
            )}
          </div>

          {/* Bottom Execution Log Terminal */}
          <LogConsole logs={currentRun.logs} />
        </div>
      </div>

      {/* Run History Comparison Modal */}
      {isHistoryOpen && (
        <RunHistoryView
          runs={historyRuns}
          onSelectRun={setCurrentRun}
          onClose={() => setIsHistoryOpen(false)}
        />
      )}
    </div>
  );
};
