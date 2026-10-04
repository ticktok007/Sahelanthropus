// src/components/neural/Stage5PPO.tsx
import React from 'react';
import { BrainCircuit, CheckCircle2, ShieldAlert, Sparkles, HelpCircle, ArrowUpRight } from 'lucide-react';
import { PPONeuralState } from '../../types/superoptimizer';

interface Stage5PPOProps {
  ppo: PPONeuralState;
  explainMode?: boolean;
}

export const Stage5PPO: React.FC<Stage5PPOProps> = ({ ppo, explainMode = false }) => {
  return (
    <div className="space-y-4">
      {explainMode && (
        <div className="bg-[#172231] border border-[#00D9FF]/40 p-3 rounded-xl text-xs text-[#E6EDF3] flex items-start space-x-2">
          <HelpCircle className="w-4 h-4 text-[#00D9FF] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-[#00D9FF]">What is PPO Policy Inference?</span>
            <p className="text-[#8B9AAF] mt-0.5">
              The Proximal Policy Optimization (PPO) agent takes the 256-D GAT embedding, applies an <strong>Action Mask</strong> to filter out invalid moves, computes Softmax policy probabilities across 160 action slots (10 rules × 16 instruction slots), and estimates the state value V(s).
            </p>
          </div>
        </div>
      )}

      {/* PPO Policy Metrics Bar */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Critic Value V(s)</span>
          <div className="text-xl font-bold font-mono text-[#36D399]">
            {ppo.criticValue > 0 ? `+${ppo.criticValue.toFixed(4)}` : ppo.criticValue.toFixed(4)}
          </div>
          <span className="text-[10px] text-[#8B9AAF]/70">Expected Return Baseline</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Active Candidates</span>
          <div className="text-xl font-bold font-mono text-[#00D9FF]">{ppo.actions.length}</div>
          <span className="text-[10px] text-[#8B9AAF]/70">Valid Unmasked Actions</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Action Space</span>
          <div className="text-xl font-bold font-mono text-[#9B7CFF]">160 Slots</div>
          <span className="text-[10px] text-[#8B9AAF]/70">10 Rules × 16 Positions</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Policy Entropy</span>
          <div className="text-xl font-bold font-mono text-[#FFB84D]">{ppo.policyEntropy.toFixed(3)}</div>
          <span className="text-[10px] text-[#8B9AAF]/70">Exploration Entropy</span>
        </div>
      </div>

      {/* Main Policy Probabilities Bar Chart & Action Masking */}
      <div className="grid grid-cols-3 gap-4">
        {/* Left Action Probabilities Bar Chart */}
        <div className="col-span-2 bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-[#263445] pb-2">
            <div className="flex items-center space-x-2">
              <BrainCircuit className="w-4 h-4 text-[#00D9FF]" />
              <h3 className="text-xs font-bold text-[#E6EDF3]">
                PPO POLICY ACTION PROBABILITY DISTRIBUTION
              </h3>
            </div>
            <span className="text-[10px] font-mono text-[#00D9FF]">SOFTMAX LOGITS</span>
          </div>

          <div className="space-y-3">
            {ppo.actions.map((act) => (
              <div key={act.actionIndex} className="bg-[#070B12] p-3 rounded-lg border border-[#263445] space-y-2">
                <div className="flex justify-between items-center text-xs font-mono">
                  <div className="flex items-center space-x-2">
                    <span className="text-[#00D9FF] font-bold">Action #{act.actionIndex}</span>
                    <span className="text-[#E6EDF3]">[Rule {act.ruleId}: {act.ruleName}]</span>
                    <span className="text-[#8B9AAF]">@ Slot {act.slotId}</span>
                  </div>
                  <span className="text-[#36D399] font-bold text-sm">
                    {(act.probability * 100).toFixed(2)}%
                  </span>
                </div>

                {/* Probability Progress Bar */}
                <div className="w-full bg-[#111A26] h-3 rounded-full overflow-hidden border border-[#263445]">
                  <div
                    className="bg-gradient-to-r from-[#00D9FF] via-[#4D8DFF] to-[#36D399] h-full rounded-full transition-all duration-500"
                    style={{ width: `${act.probability * 100}%` }}
                  />
                </div>

                <div className="flex justify-between items-center text-[11px] font-mono text-[#8B9AAF]">
                  <span>Target Candidate: <code className="text-[#FFB84D]">{act.candidateText}</code></span>
                  <span className="text-[#36D399] font-bold">SELECTED BY POLICY</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Action Mask Visualizer & Policy Decision Explanation */}
        <div className="space-y-4">
          <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-[#263445] pb-2">
              <span className="text-xs font-bold text-[#E6EDF3]">ACTION MASK STATUS</span>
              <span className="text-[10px] font-mono text-[#00D9FF]">160 SLOTS</span>
            </div>

            <div className="grid grid-cols-10 gap-1 p-2 bg-[#070B12] rounded-lg border border-[#263445]">
              {Array.from({ length: 40 }).map((_, i) => {
                const isSelected = ppo.actions.some((a) => a.actionIndex === i);
                return (
                  <div
                    key={i}
                    title={`Action Slot ${i}`}
                    className={`h-4 rounded-[2px] transition-colors ${
                      isSelected
                        ? 'bg-[#36D399] shadow-sm shadow-[#36D399]/40'
                        : 'bg-[#111A26]'
                    }`}
                  />
                );
              })}
            </div>

            <div className="flex items-center justify-between text-[10px] text-[#8B9AAF] font-mono">
              <span className="flex items-center space-x-1">
                <span className="w-2.5 h-2.5 bg-[#36D399] rounded-[2px]" />
                <span>Valid Action</span>
              </span>
              <span className="flex items-center space-x-1">
                <span className="w-2.5 h-2.5 bg-[#111A26] rounded-[2px]" />
                <span>Masked (-inf)</span>
              </span>
            </div>
          </div>

          {/* Decision Reasoning Card */}
          <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-2 text-xs">
            <div className="flex items-center space-x-1.5 text-[#00D9FF] font-bold border-b border-[#263445] pb-2">
              <Sparkles className="w-4 h-4" />
              <span>POLICY DECISION SUMMARY</span>
            </div>
            <p className="text-[#8B9AAF] leading-relaxed text-[11px]">
              The policy selected <strong>Rule 0 (mul_power2_to_slli)</strong> with high confidence (58.6%) because converting multiplication by 4 to logical shift left saves ~3 cycles, maximizing policy reward under GAE advantage estimates.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
