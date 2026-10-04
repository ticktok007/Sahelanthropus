// src/components/layout/LogConsole.tsx
import React, { useState } from 'react';
import { Terminal, ChevronUp, ChevronDown, Trash2, CheckCircle2 } from 'lucide-react';

interface LogConsoleProps {
  logs: string[];
}

export const LogConsole: React.FC<LogConsoleProps> = ({ logs }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  return (
    <div className="border-t border-[#263445] bg-[#0D131D] text-xs font-mono select-none">
      {/* Console Header Bar */}
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="px-4 py-2 flex items-center justify-between cursor-pointer hover:bg-[#111A26] transition-all"
      >
        <div className="flex items-center space-x-2">
          <Terminal className="w-3.5 h-3.5 text-[#00D9FF]" />
          <span className="font-bold text-[#E6EDF3] tracking-wide">SYSTEM EXECUTION LOGS</span>
          <span className="text-[10px] text-[#36D399] bg-[#36D399]/10 px-1.5 py-0.5 rounded border border-[#36D399]/30">
            {logs.length} EVENTS LOGGED
          </span>
        </div>

        <div className="flex items-center space-x-2 text-[#8B9AAF]">
          <span className="text-[10px]">Click to toggle terminal</span>
          {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </div>
      </div>

      {/* Console Output Body */}
      {isOpen && (
        <div className="h-44 bg-[#070B12] p-3 overflow-y-auto font-mono text-xs border-t border-[#263445] text-[#8B9AAF] space-y-1">
          {logs.map((log, idx) => (
            <div key={idx} className="hover:bg-[#172231]/40 px-2 py-0.5 rounded flex items-start space-x-2">
              <span className="text-[#00D9FF] shrink-0 font-bold">&gt;</span>
              <span className="text-[#E6EDF3]">{log}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
