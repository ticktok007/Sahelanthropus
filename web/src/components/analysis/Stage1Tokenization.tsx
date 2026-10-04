// src/components/analysis/Stage1Tokenization.tsx
import React, { useState } from 'react';
import { Code2, Eye, Sliders, Layers, HelpCircle } from 'lucide-react';
import { InstructionToken, FeatureComposition } from '../../types/superoptimizer';

interface Stage1TokenizationProps {
  tokens: InstructionToken[];
  features: FeatureComposition;
  explainMode?: boolean;
}

export const Stage1Tokenization: React.FC<Stage1TokenizationProps> = ({
  tokens,
  features,
  explainMode = false,
}) => {
  const [viewMode, setViewMode] = useState<'RAW' | 'NORMALIZED' | 'BINARY' | 'HEATMAP'>('HEATMAP');
  const [selectedSlot, setSelectedSlot] = useState<number>(0);

  const selectedToken = tokens[selectedSlot] || tokens[0];

  return (
    <div className="space-y-4">
      {explainMode && (
        <div className="bg-[#172231] border border-[#36D399]/40 p-3 rounded-xl text-xs text-[#E6EDF3] flex items-start space-x-2">
          <HelpCircle className="w-4 h-4 text-[#36D399] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-[#36D399]">What happens in Stage 1?</span>
            <p className="text-[#8B9AAF] mt-0.5">
              The assembly tokenizer converts text mnemonics into standardized opcode/register integer tuples, and expands each instruction into a high-dimensional <strong>177-dimensional vector</strong> (47 opcode bits + 96 register bits + 32 immediate bits) ready for graph neural network encoding.
            </p>
          </div>
        </div>
      )}

      {/* Feature Dimension Breakdown Overview */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Opcode One-Hot</span>
          <div className="text-xl font-bold font-mono text-[#00D9FF]">47-D</div>
          <span className="text-[10px] text-[#8B9AAF]/70">RISC-V Opcode Map</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Register Bitfields</span>
          <div className="text-xl font-bold font-mono text-[#FFB84D]">96-D</div>
          <span className="text-[10px] text-[#8B9AAF]/70">RD, RS1, RS2 (32×3)</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Immediate Vector</span>
          <div className="text-xl font-bold font-mono text-[#9B7CFF]">32-D</div>
          <span className="text-[10px] text-[#8B9AAF]/70">32-Bit Scalar Immediate</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Total Space</span>
          <div className="text-xl font-bold font-mono text-[#36D399]">177-D</div>
          <span className="text-[10px] text-[#8B9AAF]/70">Per Node Embedding</span>
        </div>
      </div>

      {/* Main Feature Matrix Visualizer */}
      <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Code2 className="w-4 h-4 text-[#00D9FF]" />
            <h3 className="text-xs font-bold text-[#E6EDF3] tracking-wide">
              177-DIMENSIONAL FEATURE MATRIX SCANNER
            </h3>
          </div>

          <div className="flex items-center space-x-1 bg-[#0D131D] p-1 rounded-lg border border-[#263445] text-[11px] font-mono">
            {(['RAW', 'NORMALIZED', 'BINARY', 'HEATMAP'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-2 py-0.5 rounded ${
                  viewMode === mode
                    ? 'bg-[#172231] text-[#00D9FF] font-bold border border-[#00D9FF]/30'
                    : 'text-[#8B9AAF] hover:text-[#E6EDF3]'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>

        {/* Feature Heatmap Matrix Grid */}
        <div className="bg-[#070B12] p-3 rounded-lg border border-[#263445] font-mono text-xs overflow-x-auto">
          <div className="space-y-2">
            {tokens.map((t, idx) => (
              <div
                key={t.slot}
                onClick={() => setSelectedSlot(t.slot)}
                className={`p-2 rounded-md transition-all cursor-pointer border ${
                  selectedSlot === t.slot
                    ? 'bg-[#172231] border-[#00D9FF]'
                    : 'bg-[#0D131D] border-[#263445] hover:border-[#8B9AAF]/50'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5 text-[11px]">
                  <span className="text-[#E6EDF3] font-bold">
                    Slot {t.slot}: <span className="text-[#00D9FF]">{t.mnemonic}</span>
                  </span>
                  <span className="text-[#8B9AAF] text-[10px]">
                    Internal: [{t.opcodeId}, {t.rs1}, {t.rs2}, {t.rd}, {t.imm}]
                  </span>
                </div>

                {/* 177-dim Heatmap Cells preview */}
                <div className="grid grid-cols-59 gap-0.5 h-4">
                  {t.featureVector.slice(0, 59).map((val, i) => (
                    <div
                      key={i}
                      title={`Dim ${i}: ${val}`}
                      className={`h-full rounded-[1px] transition-colors ${
                        val > 0
                          ? i < 47
                            ? 'bg-[#00D9FF]'
                            : i < 143
                            ? 'bg-[#FFB84D]'
                            : 'bg-[#9B7CFF]'
                          : 'bg-[#111A26]'
                      }`}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Slot Detailed View */}
        {selectedToken && (
          <div className="bg-[#0D131D] p-3 rounded-lg border border-[#263445] text-xs font-mono space-y-2">
            <span className="text-[#8B9AAF] text-[11px] block font-sans font-semibold">
              Slot {selectedToken.slot} Feature Decomposition:
            </span>
            <div className="grid grid-cols-4 gap-2 text-[11px]">
              <div>
                <span className="text-[#8B9AAF]">Opcode:</span>{' '}
                <strong className="text-[#00D9FF]">{selectedToken.opcode}</strong> ({selectedToken.opcodeId})
              </div>
              <div>
                <span className="text-[#8B9AAF]">RD:</span>{' '}
                <strong className="text-[#FFB84D]">{selectedToken.rdName}</strong> (x{selectedToken.rd})
              </div>
              <div>
                <span className="text-[#8B9AAF]">RS1 / RS2:</span>{' '}
                <strong className="text-[#FFB84D]">{selectedToken.rs1Name}</strong> /{' '}
                <strong className="text-[#FFB84D]">{selectedToken.rs2Name}</strong>
              </div>
              <div>
                <span className="text-[#8B9AAF]">Immediate:</span>{' '}
                <strong className="text-[#9B7CFF]">{selectedToken.imm}</strong>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
