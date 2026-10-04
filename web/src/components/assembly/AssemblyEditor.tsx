// src/components/assembly/AssemblyEditor.tsx
import React, { useState } from 'react';
import { Play, Sparkles, Trash2, CheckCircle2, AlertTriangle, FileCode } from 'lucide-react';
import { BenchmarkExample } from '../../types/superoptimizer';
import { BENCHMARK_EXAMPLES } from '../../services/mockService';

interface AssemblyEditorProps {
  code: string;
  onChange: (newCode: string) => void;
  onRun: () => void;
  onSelectExample: (ex: BenchmarkExample) => void;
}

export const AssemblyEditor: React.FC<AssemblyEditorProps> = ({
  code,
  onChange,
  onRun,
  onSelectExample,
}) => {
  const [selectedExampleId, setSelectedExampleId] = useState<string>('ex5');

  const lines = code.split('\n');

  const handleSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    setSelectedExampleId(id);
    const ex = BENCHMARK_EXAMPLES.find((item) => item.id === id);
    if (ex) {
      onSelectExample(ex);
    }
  };

  const handleClear = () => {
    onChange('');
  };

  const handleFormat = () => {
    const formatted = code
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 0)
      .join('\n');
    onChange(formatted);
  };

  return (
    <div className="bg-[#111A26] border border-[#263445] rounded-xl flex flex-col h-full overflow-hidden shadow-xl">
      {/* Editor Toolbar */}
      <div className="bg-[#0D131D] px-4 py-2.5 border-b border-[#263445] flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-1.5 text-xs font-semibold text-[#E6EDF3]">
            <FileCode className="w-4 h-4 text-[#00D9FF]" />
            <span>RISC-V ASSEMBLY INPUT WORKSPACE</span>
          </div>

          <span className="text-[#263445]">|</span>

          {/* Preset Example Loader */}
          <div className="flex items-center space-x-2">
            <span className="text-[11px] text-[#8B9AAF]">Preset Examples:</span>
            <select
              value={selectedExampleId}
              onChange={handleSelect}
              className="bg-[#172231] border border-[#263445] text-[#00D9FF] text-xs font-mono rounded-lg px-2.5 py-1 focus:outline-none focus:border-[#00D9FF]"
            >
              {BENCHMARK_EXAMPLES.map((ex) => (
                <option key={ex.id} value={ex.id}>
                  {ex.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Editor Controls */}
        <div className="flex items-center space-x-2 text-xs">
          <button
            onClick={handleFormat}
            className="px-2.5 py-1 rounded bg-[#172231] border border-[#263445] text-[#8B9AAF] hover:text-[#E6EDF3] transition-all"
          >
            Format
          </button>
          <button
            onClick={handleClear}
            className="px-2.5 py-1 rounded bg-[#172231] border border-[#263445] text-[#FF5C6C]/80 hover:text-[#FF5C6C] transition-all flex items-center space-x-1"
          >
            <Trash2 className="w-3 h-3" />
            <span>Clear</span>
          </button>
          <button
            onClick={onRun}
            className="bg-[#00D9FF] hover:bg-[#00D9FF]/90 text-[#070B12] font-bold px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition-all shadow-md shadow-[#00D9FF]/20 active:scale-95"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>SUPEROPTIMIZE</span>
          </button>
        </div>
      </div>

      {/* Editor Body with Line Numbers */}
      <div className="flex-1 flex overflow-hidden relative bg-[#070B12] font-mono text-xs">
        {/* Line Numbers Column */}
        <div className="w-12 bg-[#0D131D] text-[#8B9AAF]/40 py-3 text-right pr-3 select-none border-r border-[#263445]/50 flex flex-col space-y-1">
          {lines.map((_, i) => (
            <div key={i} className="h-5 leading-5 text-[11px]">
              {i + 1}
            </div>
          ))}
        </div>

        {/* Assembly Code Text Area */}
        <textarea
          value={code}
          onChange={(e) => onChange(e.target.value)}
          placeholder="// Paste or write RISC-V assembly here... (e.g. mul t1, a0, t0)"
          spellCheck={false}
          className="w-full h-full bg-transparent text-[#E6EDF3] p-3 resize-none focus:outline-none leading-5 font-mono text-xs selection:bg-[#00D9FF]/30"
        />
      </div>

      {/* Bottom Status bar */}
      <div className="bg-[#0D131D] px-4 py-1.5 border-t border-[#263445] flex items-center justify-between text-[11px] font-mono text-[#8B9AAF]">
        <div className="flex items-center space-x-3">
          <span className="flex items-center space-x-1 text-[#36D399]">
            <CheckCircle2 className="w-3 h-3" />
            <span>Syntax Valid (RV32I)</span>
          </span>
          <span>•</span>
          <span>{lines.length} lines</span>
        </div>
        <span>Architecture: RV32I / RISC-V 32-Bit Base</span>
      </div>
    </div>
  );
};
