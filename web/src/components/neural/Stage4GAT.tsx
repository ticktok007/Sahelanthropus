// src/components/neural/Stage4GAT.tsx
import React, { useState } from 'react';
import { Layers, Brain, Eye, HelpCircle, Network } from 'lucide-react';
import { GATState } from '../../types/superoptimizer';

interface Stage4GATProps {
  gat: GATState;
  explainMode?: boolean;
}

export const Stage4GAT: React.FC<Stage4GATProps> = ({ gat, explainMode = false }) => {
  const [selectedAttentionIndex, setSelectedAttentionIndex] = useState<number>(0);

  const selectedAtt = gat.attentionEdges[selectedAttentionIndex] || gat.attentionEdges[0];

  return (
    <div className="space-y-4">
      {explainMode && (
        <div className="bg-[#172231] border border-[#9B7CFF]/40 p-3 rounded-xl text-xs text-[#E6EDF3] flex items-start space-x-2">
          <HelpCircle className="w-4 h-4 text-[#9B7CFF] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-[#9B7CFF]">How does HeteroGAT work?</span>
            <p className="text-[#8B9AAF] mt-0.5">
              The Heterogeneous Graph Attention Network (HeteroGAT) processes the assembly program graph over 3 GraphConv layers using 8 multi-head attention mechanisms. It computes graph-level representations of control flow and def-use chains, producing a <strong>256-dimensional embedding vector</strong>.
            </p>
          </div>
        </div>
      )}

      {/* Network Hyperparameter Metrics */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Input Channels</span>
          <div className="text-xl font-bold font-mono text-[#00D9FF]">{gat.instInDim}-D</div>
          <span className="text-[10px] text-[#8B9AAF]/70">Node Features</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Attention Heads</span>
          <div className="text-xl font-bold font-mono text-[#9B7CFF]">{gat.numHeads} Heads</div>
          <span className="text-[10px] text-[#8B9AAF]/70">Multi-Head GAT</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Conv Layers</span>
          <div className="text-xl font-bold font-mono text-[#FFB84D]">{gat.numLayers} Layers</div>
          <span className="text-[10px] text-[#8B9AAF]/70">HeteroConv Stack</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Graph Embedding</span>
          <div className="text-xl font-bold font-mono text-[#36D399]">{gat.embedDim}-D</div>
          <span className="text-[10px] text-[#8B9AAF]/70">Global Pool Vector</span>
        </div>
      </div>

      {/* Main Architecture & Attention Visualization */}
      <div className="grid grid-cols-2 gap-4">
        {/* Left Neural Architecture Flow */}
        <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
          <div className="flex items-center space-x-2 border-b border-[#263445] pb-2">
            <Layers className="w-4 h-4 text-[#9B7CFF]" />
            <h3 className="text-xs font-bold text-[#E6EDF3]">HETERO-GAT BACKBONE ARCHITECTURE</h3>
          </div>

          <div className="space-y-2 font-mono text-xs">
            <div className="bg-[#070B12] p-2.5 rounded-lg border border-[#263445] flex justify-between items-center">
              <div>
                <span className="text-[#00D9FF] font-bold">1. Program Graph Input</span>
                <span className="text-[10px] text-[#8B9AAF] block">HeteroData (inst + reg nodes)</span>
              </div>
              <span className="text-xs text-[#8B9AAF]">177-D</span>
            </div>

            <div className="text-center text-[#9B7CFF]">↓</div>

            <div className="bg-[#070B12] p-2.5 rounded-lg border border-[#263445] flex justify-between items-center">
              <div>
                <span className="text-[#9B7CFF] font-bold">2. HeteroConv Layer 1</span>
                <span className="text-[10px] text-[#8B9AAF] block">8-head GAT edge aggregation</span>
              </div>
              <span className="text-xs text-[#8B9AAF]">128-D</span>
            </div>

            <div className="text-center text-[#9B7CFF]">↓</div>

            <div className="bg-[#070B12] p-2.5 rounded-lg border border-[#263445] flex justify-between items-center">
              <div>
                <span className="text-[#9B7CFF] font-bold">3. HeteroConv Layer 2 & 3</span>
                <span className="text-[10px] text-[#8B9AAF] block">Deep contextual message passing</span>
              </div>
              <span className="text-xs text-[#8B9AAF]">128-D</span>
            </div>

            <div className="text-center text-[#9B7CFF]">↓</div>

            <div className="bg-[#070B12] p-2.5 rounded-lg border border-[#36D399]/40 flex justify-between items-center">
              <div>
                <span className="text-[#36D399] font-bold">4. Global Pooling Projection</span>
                <span className="text-[10px] text-[#8B9AAF] block">Mean + max graph readout embedding</span>
              </div>
              <span className="text-xs text-[#36D399] font-bold">256-D</span>
            </div>
          </div>
        </div>

        {/* Right GAT Attention Weights Distribution */}
        <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-[#263445] pb-2">
            <div className="flex items-center space-x-2">
              <Brain className="w-4 h-4 text-[#00D9FF]" />
              <h3 className="text-xs font-bold text-[#E6EDF3]">ATTENTION WEIGHT DISTRIBUTION</h3>
            </div>
            <span className="text-[10px] font-mono text-[#9B7CFF]">LAYER 3 HEADS</span>
          </div>

          <div className="space-y-2">
            {gat.attentionEdges.map((att, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedAttentionIndex(idx)}
                className={`p-2.5 rounded-lg border cursor-pointer transition-all ${
                  selectedAttentionIndex === idx
                    ? 'bg-[#172231] border-[#9B7CFF]'
                    : 'bg-[#070B12] border-[#263445] hover:border-[#8B9AAF]/50'
                }`}
              >
                <div className="flex justify-between items-center font-mono text-xs mb-1">
                  <span className="text-[#E6EDF3] font-bold">
                    {att.sourceNodeId} <span className="text-[#9B7CFF]">➔</span> {att.targetNodeId}
                  </span>
                  <span className="text-[#36D399] font-bold">
                    {(att.weight * 100).toFixed(1)}% weight
                  </span>
                </div>
                <div className="w-full bg-[#111A26] h-2 rounded-full overflow-hidden border border-[#263445] mb-1">
                  <div
                    className="bg-gradient-to-r from-[#00D9FF] to-[#9B7CFF] h-full rounded-full"
                    style={{ width: `${att.weight * 100}%` }}
                  />
                </div>
                <p className="text-[10px] text-[#8B9AAF] font-sans">{att.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
