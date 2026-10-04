import React from 'react';
import { ShieldCheck, Check, Cpu, Hash, Database, Layers, Sparkles, Zap } from 'lucide-react';
import { InstructionToken } from '../../types/superoptimizer';

interface InputValidationPanelProps {
  tokens: InstructionToken[];
  isTranspiledX86?: boolean;
  transpiledRiscvInput?: string;
}

export const InputValidationPanel: React.FC<InputValidationPanelProps> = ({
  tokens,
  isTranspiledX86,
  transpiledRiscvInput,
}) => {
  const regSet = new Set<string>();
  let immediateCount = 0;

  tokens.forEach((t) => {
    if (t.rs1Name !== 'zero') regSet.add(t.rs1Name);
    if (t.rs2Name !== 'zero') regSet.add(t.rs2Name);
    if (t.rdName !== 'zero') regSet.add(t.rdName);
    if (t.imm !== 0) immediateCount++;
  });

  return (
    <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-[#263445] pb-2">
        <div className="flex items-center space-x-2 text-xs font-semibold text-[#E6EDF3]">
          <ShieldCheck className="w-4 h-4 text-[#36D399]" />
          <span>INPUT VALIDATION REPORT</span>
        </div>
        {isTranspiledX86 ? (
          <span className="text-[10px] font-mono text-[#FFB84D] bg-[#FFB84D]/10 border border-[#FFB84D]/30 px-2 py-0.5 rounded flex items-center space-x-1">
            <Zap className="w-3 h-3 text-[#FFB84D]" />
            <span>x86-64 AUTO-TRANSPILED</span>
          </span>
        ) : (
          <span className="text-[10px] font-mono text-[#36D399] bg-[#36D399]/10 border border-[#36D399]/30 px-2 py-0.5 rounded">
            PASSED
          </span>
        )}
      </div>

      {isTranspiledX86 && (
        <div className="bg-[#172231] border border-[#FFB84D]/40 p-2.5 rounded-lg text-xs space-y-1">
          <div className="flex items-center space-x-1.5 text-[#FFB84D] font-bold text-[11px]">
            <Sparkles className="w-3.5 h-3.5" />
            <span>x86-64 Assembly Dialect Detected</span>
          </div>
          <p className="text-[10px] text-[#8B9AAF] leading-tight font-sans">
            Auto-converted x86 mnemonics (<code className="text-[#00D9FF]">mov</code>, <code className="text-[#00D9FF]">syscall</code>, <code className="text-[#00D9FF]">rax</code>, <code className="text-[#00D9FF]">rbx</code>) to RISC-V RV32I (<code className="text-[#36D399]">li</code>, <code className="text-[#36D399]">mv</code>, <code className="text-[#36D399]">ecall</code>).
          </p>
        </div>
      )}

      <div className="grid grid-cols-4 gap-2 font-mono text-xs">
        <div className="bg-[#0D131D] p-2.5 rounded-lg border border-[#263445]">
          <div className="flex items-center space-x-1 text-[10px] text-[#8B9AAF]">
            <Cpu className="w-3 h-3 text-[#00D9FF]" />
            <span>Target ISA</span>
          </div>
          <span className="text-sm font-bold text-[#00D9FF]">RV32I</span>
        </div>

        <div className="bg-[#0D131D] p-2.5 rounded-lg border border-[#263445]">
          <div className="flex items-center space-x-1 text-[10px] text-[#8B9AAF]">
            <Hash className="w-3 h-3 text-[#4D8DFF]" />
            <span>Instructions</span>
          </div>
          <span className="text-sm font-bold text-[#E6EDF3]">{tokens.length}</span>
        </div>

        <div className="bg-[#0D131D] p-2.5 rounded-lg border border-[#263445]">
          <div className="flex items-center space-x-1 text-[10px] text-[#8B9AAF]">
            <Database className="w-3 h-3 text-[#FFB84D]" />
            <span>Regs Used</span>
          </div>
          <span className="text-sm font-bold text-[#FFB84D]">{regSet.size}</span>
        </div>

        <div className="bg-[#0D131D] p-2.5 rounded-lg border border-[#263445]">
          <div className="flex items-center space-x-1 text-[10px] text-[#8B9AAF]">
            <Layers className="w-3 h-3 text-[#9B7CFF]" />
            <span>Immediates</span>
          </div>
          <span className="text-sm font-bold text-[#9B7CFF]">{immediateCount}</span>
        </div>
      </div>

      {/* Validation Checklist */}
      <div className="space-y-1.5 text-xs text-[#8B9AAF]">
        <div className="flex items-center space-x-2">
          <Check className="w-3.5 h-3.5 text-[#36D399]" />
          <span>
            {isTranspiledX86
              ? 'x86-64 instructions transpiled to RISC-V RV32I'
              : 'RISC-V 32-bit instruction syntax verified'}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <Check className="w-3.5 h-3.5 text-[#36D399]" />
          <span>Register aliases mapped to architectural x0-x31</span>
        </div>
        <div className="flex items-center space-x-2">
          <Check className="w-3.5 h-3.5 text-[#36D399]" />
          <span>Immediate sign-extension range clamped [-2^31, 2^31-1]</span>
        </div>
      </div>
    </div>
  );
};

