// src/components/pipeline/Stage3Peephole.tsx
import React, { useState } from 'react';
import { Binary, CheckCircle2, Zap, HelpCircle, ArrowRight } from 'lucide-react';
import { PeepholeRule, RuleMatchCell, InstructionToken } from '../../types/superoptimizer';

interface Stage3PeepholeProps {
  rules: PeepholeRule[];
  matches: RuleMatchCell[];
  tokens: InstructionToken[];
  explainMode?: boolean;
}

export const Stage3Peephole: React.FC<Stage3PeepholeProps> = ({
  rules,
  matches,
  tokens,
  explainMode = false,
}) => {
  const [selectedRuleId, setSelectedRuleId] = useState<number>(0);

  const selectedRule = rules.find((r) => r.id === selectedRuleId);

  return (
    <div className="space-y-4">
      {explainMode && (
        <div className="bg-[#172231] border border-[#FFB84D]/40 p-3 rounded-xl text-xs text-[#E6EDF3] flex items-start space-x-2">
          <HelpCircle className="w-4 h-4 text-[#FFB84D] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-[#FFB84D]">What is the Peephole Rulebook?</span>
            <p className="text-[#8B9AAF] mt-0.5">
              The rulebook contains 10 algebraic rewrite patterns (e.g. converting multiplication by power of 2 into left shifts, or cancelling out inverse operations). The scanner evaluates each instruction slot to generate valid optimization candidates for the neural policy.
            </p>
          </div>
        </div>
      )}

      {/* Rules Match Matrix Grid */}
      <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-[#263445] pb-2">
          <div className="flex items-center space-x-2">
            <Binary className="w-4 h-4 text-[#00D9FF]" />
            <h3 className="text-xs font-bold text-[#E6EDF3]">
              PEEPHOLE RULEBOOK MATCH MATRIX (N_rules = 10)
            </h3>
          </div>
          <span className="text-[10px] font-mono text-[#36D399] bg-[#36D399]/10 border border-[#36D399]/30 px-2 py-0.5 rounded">
            {matches.length} PATTERNS MATCHED
          </span>
        </div>

        {/* Heatmap Grid Matrix */}
        <div className="overflow-x-auto bg-[#070B12] p-3 rounded-lg border border-[#263445] font-mono text-xs">
          <table className="w-full text-left">
            <thead>
              <tr className="text-[#8B9AAF] border-b border-[#263445] text-[10px]">
                <th className="pb-2">Instruction Slot</th>
                {rules.map((r) => (
                  <th key={r.id} className="pb-2 text-center px-1">
                    R{r.id}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-[#263445]/50 text-[11px]">
              {tokens.map((t) => (
                <tr key={t.slot} className="hover:bg-[#172231]/50">
                  <td className="py-2 text-[#E6EDF3] font-bold">
                    Slot {t.slot}: <span className="text-[#00D9FF]">{t.mnemonic}</span>
                  </td>
                  {rules.map((r) => {
                    const match = matches.find((m) => m.slot === t.slot && m.ruleId === r.id);
                    return (
                      <td key={r.id} className="py-2 text-center">
                        <div
                          onClick={() => setSelectedRuleId(r.id)}
                          className={`w-6 h-6 mx-auto rounded flex items-center justify-center cursor-pointer transition-transform hover:scale-110 font-bold ${
                            match
                              ? 'bg-[#36D399] text-[#070B12] shadow-sm shadow-[#36D399]/40'
                              : 'bg-[#111A26] text-[#8B9AAF]/30 hover:text-[#8B9AAF]'
                          }`}
                        >
                          {match ? '✓' : '•'}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 10 Rule Explorer Cards Grid */}
      <div className="grid grid-cols-2 gap-3">
        {rules.map((r) => {
          const matchedCells = matches.filter((m) => m.ruleId === r.id);
          const isMatched = matchedCells.length > 0;

          return (
            <div
              key={r.id}
              onClick={() => setSelectedRuleId(r.id)}
              className={`bg-[#111A26] border rounded-xl p-3 space-y-2 cursor-pointer transition-all ${
                selectedRuleId === r.id
                  ? 'border-[#00D9FF] bg-[#172231]/80 shadow-md'
                  : 'border-[#263445] hover:border-[#8B9AAF]/50'
              }`}
            >
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-[#00D9FF] font-bold">Rule {r.id}</span>
                  <span className="text-[#E6EDF3] font-semibold">{r.name}</span>
                </div>
                {isMatched ? (
                  <span className="text-[10px] font-mono text-[#36D399] bg-[#36D399]/10 px-2 py-0.5 rounded border border-[#36D399]/30">
                    MATCHED ({matchedCells.length})
                  </span>
                ) : (
                  <span className="text-[10px] font-mono text-[#8B9AAF] bg-[#0D131D] px-2 py-0.5 rounded border border-[#263445]">
                    NO MATCH
                  </span>
                )}
              </div>

              <div className="bg-[#070B12] p-2 rounded border border-[#263445] font-mono text-xs flex items-center justify-between">
                <span className="text-[#FFB84D]">{r.pattern}</span>
                <ArrowRight className="w-3.5 h-3.5 text-[#8B9AAF]" />
                <span className="text-[#36D399]">{r.replacement}</span>
              </div>

              <div className="flex justify-between items-center text-[10px] text-[#8B9AAF]">
                <span>{r.desc}</span>
                <span className="text-[#9B7CFF] font-mono font-bold">
                  ~{r.estSavings} cycles saved
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
