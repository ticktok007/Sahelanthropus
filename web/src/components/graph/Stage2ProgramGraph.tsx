// src/components/graph/Stage2ProgramGraph.tsx
import React, { useState } from 'react';
import { Network, Database, GitBranch, Info, Filter, ZoomIn, ZoomOut } from 'lucide-react';
import { GraphData, GraphNode, GraphEdge } from '../../types/superoptimizer';

interface Stage2ProgramGraphProps {
  graph: GraphData;
  explainMode?: boolean;
}

export const Stage2ProgramGraph: React.FC<Stage2ProgramGraphProps> = ({
  graph,
  explainMode = false,
}) => {
  const [selectedNodeId, setSelectedNodeId] = useState<string>(
    graph.nodes.length > 0 ? graph.nodes[0].id : ''
  );
  const [edgeFilter, setEdgeFilter] = useState<'ALL' | 'CONTROL' | 'DATA_FLOW'>('ALL');

  const selectedNode = graph.nodes.find((n) => n.id === selectedNodeId);

  const filteredEdges = graph.edges.filter((e) => {
    if (edgeFilter === 'CONTROL') return e.type === 'control_flow';
    if (edgeFilter === 'DATA_FLOW') return e.type === 'data_flow';
    return true;
  });

  return (
    <div className="space-y-4">
      {explainMode && (
        <div className="bg-[#172231] border border-[#00D9FF]/40 p-3 rounded-xl text-xs text-[#E6EDF3] flex items-start space-x-2">
          <Info className="w-4 h-4 text-[#00D9FF] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-[#00D9FF]">Why is the Program Graph important?</span>
            <p className="text-[#8B9AAF] mt-0.5">
              Superoptimization relies on understanding data dependencies. The HeteroData Graph connects instruction definitions (destinations) to their consumption sites (sources). The neural policy uses graph structural embeddings to ensure rewrites preserve exact variable flow.
            </p>
          </div>
        </div>
      )}

      {/* Graph Metrics Summary Bar */}
      <div className="grid grid-cols-4 gap-3">
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Graph Nodes</span>
          <div className="text-xl font-bold font-mono text-[#00D9FF]">{graph.nodes.length}</div>
          <span className="text-[10px] text-[#8B9AAF]/70">Instruction Vertices</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Def-Use Edges</span>
          <div className="text-xl font-bold font-mono text-[#4D8DFF]">
            {graph.edges.filter((e) => e.type === 'data_flow').length}
          </div>
          <span className="text-[10px] text-[#8B9AAF]/70">Data Flow Dependency</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Graph Density</span>
          <div className="text-xl font-bold font-mono text-[#9B7CFF]">
            {graph.density.toFixed(2)}
          </div>
          <span className="text-[10px] text-[#8B9AAF]/70">Edges / Nodes Ratio</span>
        </div>
        <div className="bg-[#111A26] border border-[#263445] p-3 rounded-xl">
          <span className="text-[11px] text-[#8B9AAF]">Max Dep Depth</span>
          <div className="text-xl font-bold font-mono text-[#36D399]">
            {graph.maxDepth}
          </div>
          <span className="text-[10px] text-[#8B9AAF]/70">Critical Path Length</span>
        </div>
      </div>

      {/* Interactive Graph Canvas & Inspector Grid */}
      <div className="grid grid-cols-3 gap-4">
        {/* Left Interactive SVG Graph Workspace */}
        <div className="col-span-2 bg-[#111A26] border border-[#263445] rounded-xl p-4 flex flex-col h-[400px]">
          <div className="flex items-center justify-between border-b border-[#263445] pb-2 mb-3">
            <div className="flex items-center space-x-2">
              <Network className="w-4 h-4 text-[#00D9FF]" />
              <h3 className="text-xs font-bold text-[#E6EDF3]">HETERODATA PROGRAM GRAPH</h3>
            </div>

            <div className="flex items-center space-x-2 text-xs">
              <div className="flex items-center space-x-1 bg-[#0D131D] p-1 rounded-lg border border-[#263445]">
                {(['ALL', 'CONTROL', 'DATA_FLOW'] as const).map((filter) => (
                  <button
                    key={filter}
                    onClick={() => setEdgeFilter(filter)}
                    className={`px-2 py-0.5 text-[11px] font-mono rounded ${
                      edgeFilter === filter
                        ? 'bg-[#172231] text-[#00D9FF] font-bold'
                        : 'text-[#8B9AAF] hover:text-[#E6EDF3]'
                    }`}
                  >
                    {filter}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* SVG Node Layout Canvas */}
          <div className="flex-1 bg-[#070B12] rounded-lg border border-[#263445] relative overflow-y-auto max-h-[340px] p-4">
            {(() => {
              const numRows = Math.max(1, Math.ceil(graph.nodes.length / 2));
              const svgHeight = Math.max(320, numRows * 80 + 50);
              const svgWidth = 480;

              return (
                <svg
                  viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                  className="w-full"
                  style={{ minHeight: `${svgHeight}px` }}
                >
                  <defs>
                    <marker
                      id="arrow-df"
                      viewBox="0 0 10 10"
                      refX="6"
                      refY="5"
                      markerWidth="6"
                      markerHeight="6"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#00D9FF" />
                    </marker>
                    <marker
                      id="arrow-cf"
                      viewBox="0 0 10 10"
                      refX="6"
                      refY="5"
                      markerWidth="6"
                      markerHeight="6"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 0 L 10 5 L 0 10 z" fill="#263445" />
                    </marker>
                  </defs>

                  {/* Render Edge Connectors */}
                  {filteredEdges.map((e) => {
                    const srcIdx = graph.nodes.findIndex((n) => n.id === e.source);
                    const dstIdx = graph.nodes.findIndex((n) => n.id === e.target);
                    if (srcIdx < 0 || dstIdx < 0) return null;

                    const x1 = 80 + (srcIdx % 2) * 200;
                    const y1 = 45 + Math.floor(srcIdx / 2) * 80;
                    const x2 = 80 + (dstIdx % 2) * 200;
                    const y2 = 45 + Math.floor(dstIdx / 2) * 80;

                    const isDF = e.type === 'data_flow';

                    return (
                      <g key={e.id}>
                        <line
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke={isDF ? '#00D9FF' : '#263445'}
                          strokeWidth={isDF ? 2 : 1}
                          strokeDasharray={isDF ? 'none' : '4 4'}
                          markerEnd={isDF ? 'url(#arrow-df)' : 'url(#arrow-cf)'}
                        />
                        {isDF && e.reg && (
                          <text
                            x={(x1 + x2) / 2 + 8}
                            y={(y1 + y2) / 2}
                            fill="#FFB84D"
                            fontSize="10"
                            fontFamily="JetBrains Mono"
                          >
                            Def-Use: {e.reg}
                          </text>
                        )}
                      </g>
                    );
                  })}

                  {/* Render Node Circles */}
                  {graph.nodes.map((n, idx) => {
                    const cx = 80 + (idx % 2) * 200;
                    const cy = 45 + Math.floor(idx / 2) * 80;
                    const isSelected = selectedNodeId === n.id;

                    return (
                      <g
                        key={n.id}
                        onClick={() => setSelectedNodeId(n.id)}
                        className="cursor-pointer transition-transform hover:scale-105"
                      >
                        <circle
                          cx={cx}
                          cy={cy}
                          r="20"
                          fill={isSelected ? '#172231' : '#0D131D'}
                          stroke={isSelected ? '#00D9FF' : '#263445'}
                          strokeWidth={isSelected ? 3 : 1.5}
                        />
                        <text
                          x={cx}
                          y={cy + 4}
                          textAnchor="middle"
                          fill={isSelected ? '#00D9FF' : '#E6EDF3'}
                          fontSize="11"
                          fontWeight="bold"
                          fontFamily="JetBrains Mono"
                        >
                          {n.id}
                        </text>
                        <text
                          x={cx + 26}
                          y={cy + 4}
                          fill="#8B9AAF"
                          fontSize="10"
                          fontFamily="JetBrains Mono"
                        >
                          {n.label.split(':')[1]}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              );
            })()}
          </div>


          {/* Graph Legend */}
          <div className="flex items-center justify-between text-[11px] text-[#8B9AAF] font-mono mt-3">
            <div className="flex items-center space-x-4">
              <span className="flex items-center space-x-1">
                <span className="w-3 h-0.5 bg-[#00D9FF]" />
                <span>Def-Use Data Flow</span>
              </span>
              <span className="flex items-center space-x-1">
                <span className="w-3 h-0.5 bg-[#263445] border-t border-dashed" />
                <span>Control Flow</span>
              </span>
            </div>
            <span>Click any node to inspect register reads/writes</span>
          </div>
        </div>

        {/* Right Node Details & Register Usage Drawer */}
        <div className="space-y-4">
          {/* Selected Node Inspector */}
          <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-[#263445] pb-2">
              <span className="text-xs font-bold text-[#E6EDF3]">NODE INSPECTOR</span>
              {selectedNode && (
                <span className="text-[10px] font-mono text-[#00D9FF] bg-[#00D9FF]/10 px-2 py-0.5 rounded border border-[#00D9FF]/30">
                  {selectedNode.id}
                </span>
              )}
            </div>

            {selectedNode ? (
              <div className="space-y-3 font-mono text-xs">
                <div>
                  <span className="text-[10px] text-[#8B9AAF] block">Instruction Mnemonic</span>
                  <span className="text-[#00D9FF] font-bold text-sm">{selectedNode.label}</span>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="bg-[#0D131D] p-2 rounded border border-[#263445]">
                    <span className="text-[#8B9AAF] text-[10px] block">Registers Read</span>
                    <span className="text-[#FFB84D] font-bold">
                      {selectedNode.reads.length > 0 ? selectedNode.reads.join(', ') : 'None'}
                    </span>
                  </div>

                  <div className="bg-[#0D131D] p-2 rounded border border-[#263445]">
                    <span className="text-[#8B9AAF] text-[10px] block">Registers Written</span>
                    <span className="text-[#36D399] font-bold">
                      {selectedNode.writes.length > 0 ? selectedNode.writes.join(', ') : 'None'}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <span className="text-xs text-[#8B9AAF]">Select a graph node to inspect</span>
            )}
          </div>

          {/* Register Usage Bar Chart */}
          <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
            <span className="text-xs font-bold text-[#E6EDF3] block border-b border-[#263445] pb-2">
              REGISTER USAGE FREQUENCY
            </span>

            <div className="space-y-2 text-xs font-mono">
              {Object.entries(graph.registerUsage).map(([reg, count]) => (
                <div key={reg} className="space-y-1">
                  <div className="flex justify-between text-[11px]">
                    <span className="text-[#FFB84D] font-bold">{reg}</span>
                    <span className="text-[#8B9AAF]">{count} refs</span>
                  </div>
                  <div className="w-full bg-[#0D131D] h-2 rounded-full overflow-hidden border border-[#263445]">
                    <div
                      className="bg-[#00D9FF] h-full rounded-full"
                      style={{ width: `${(count / 4) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
