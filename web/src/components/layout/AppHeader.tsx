// src/components/layout/AppHeader.tsx
import React from 'react';
import { Cpu, Play, CheckCircle2, Shield, Settings, History, HelpCircle, Eye, Sparkles } from 'lucide-react';
import { UserMode } from '../../types/superoptimizer';

interface AppHeaderProps {
  runId: string;
  architecture: string;
  status: string;
  isVerified: boolean;
  userMode: UserMode;
  onModeChange: (mode: UserMode) => void;
  onRunSuperopt: () => void;
  onToggleHistory: () => void;
  isHistoryOpen: boolean;
  onGoToLanding?: () => void;
}

export const AppHeader: React.FC<AppHeaderProps> = ({
  runId,
  architecture,
  status,
  isVerified,
  userMode,
  onModeChange,
  onRunSuperopt,
  onToggleHistory,
  isHistoryOpen,
  onGoToLanding,
}) => {
  return (
    <header className="h-14 bg-[#0D131D] border-b border-[#263445] px-4 flex items-center justify-between z-20">
      {/* Left Title & System Status */}
      <div className="flex items-center space-x-4">
        <div
          onClick={onGoToLanding}
          className="flex items-center space-x-2 cursor-pointer group"
          title="Return to Home Landing Page"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#00D9FF] to-[#9B7CFF] p-0.5 flex items-center justify-center shadow-lg shadow-[#00D9FF]/20 group-hover:scale-105 transition-transform">
            <div className="w-full h-full bg-[#070B12] rounded-[6px] flex items-center justify-center">
              <Cpu className="w-4 h-4 text-[#00D9FF]" />
            </div>
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-wider text-[#E6EDF3] uppercase flex items-center gap-2 group-hover:text-[#00D9FF] transition-colors">
              SAHELANTHROPUS
              <span className="text-[10px] bg-[#172231] border border-[#263445] text-[#00D9FF] px-1.5 py-0.5 rounded font-mono font-normal">
                v1.0.4
              </span>
            </h1>
            <p className="text-[10px] text-[#8B9AAF]">RISC-V Neural Superoptimizer & Formal Verifier</p>
          </div>
        </div>


        <div className="h-6 w-[1px] bg-[#263445] mx-1" />

        {/* Current Run Metadata */}
        <div className="flex items-center space-x-3 text-xs font-mono">
          <span className="text-[#8B9AAF]">Run: <strong className="text-[#00D9FF]">{runId}</strong></span>
          <span className="text-[#263445]">|</span>
          <span className="text-[#8B9AAF]">Target: <strong className="text-[#E6EDF3]">{architecture}</strong></span>
          <span className="text-[#263445]">|</span>
          {isVerified ? (
            <span className="flex items-center space-x-1 text-[#36D399] bg-[#36D399]/10 px-2 py-0.5 rounded border border-[#36D399]/30">
              <Shield className="w-3 h-3" />
              <span>Z3 FORMALLY VERIFIED</span>
            </span>
          ) : (
            <span className="flex items-center space-x-1 text-[#FFB84D] bg-[#FFB84D]/10 px-2 py-0.5 rounded border border-[#FFB84D]/30">
              <CheckCircle2 className="w-3 h-3" />
              <span>{status}</span>
            </span>
          )}
        </div>
      </div>

      {/* Center User Mode Switcher */}
      <div className="flex items-center bg-[#111A26] border border-[#263445] rounded-lg p-0.5 text-xs font-medium">
        <button
          onClick={() => onModeChange('normal')}
          className={`px-3 py-1 rounded-md transition-all flex items-center space-x-1 ${
            userMode === 'normal' ? 'bg-[#172231] text-[#00D9FF] shadow' : 'text-[#8B9AAF] hover:text-[#E6EDF3]'
          }`}
        >
          <Eye className="w-3.5 h-3.5" />
          <span>Standard</span>
        </button>
        <button
          onClick={() => onModeChange('explain')}
          className={`px-3 py-1 rounded-md transition-all flex items-center space-x-1 ${
            userMode === 'explain' ? 'bg-[#172231] text-[#36D399] shadow' : 'text-[#8B9AAF] hover:text-[#E6EDF3]'
          }`}
        >
          <HelpCircle className="w-3.5 h-3.5" />
          <span>Explain</span>
        </button>
        <button
          onClick={() => onModeChange('research')}
          className={`px-3 py-1 rounded-md transition-all flex items-center space-x-1 ${
            userMode === 'research' ? 'bg-[#172231] text-[#9B7CFF] shadow' : 'text-[#8B9AAF] hover:text-[#E6EDF3]'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>Research</span>
        </button>
        <button
          onClick={() => onModeChange('expert')}
          className={`px-3 py-1 rounded-md transition-all flex items-center space-x-1 ${
            userMode === 'expert' ? 'bg-[#172231] text-[#FFB84D] shadow' : 'text-[#8B9AAF] hover:text-[#E6EDF3]'
          }`}
        >
          <Settings className="w-3.5 h-3.5" />
          <span>Expert</span>
        </button>
      </div>

      {/* Right Action Controls */}
      <div className="flex items-center space-x-3">
        <button
          onClick={onToggleHistory}
          className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-all flex items-center space-x-1.5 ${
            isHistoryOpen
              ? 'bg-[#172231] border-[#00D9FF] text-[#00D9FF]'
              : 'bg-[#111A26] border-[#263445] text-[#8B9AAF] hover:text-[#E6EDF3]'
          }`}
        >
          <History className="w-3.5 h-3.5" />
          <span>Runs Log</span>
        </button>

        <button
          onClick={onRunSuperopt}
          className="bg-gradient-to-r from-[#00D9FF] to-[#4D8DFF] hover:brightness-110 text-[#070B12] px-4 py-1.5 rounded-lg text-xs font-bold transition-all shadow-md shadow-[#00D9FF]/20 flex items-center space-x-1.5 active:scale-95"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>RUN SUPEROPTIMIZER</span>
        </button>
      </div>
    </header>
  );
};
