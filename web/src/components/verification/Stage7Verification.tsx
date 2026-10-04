// src/components/verification/Stage7Verification.tsx
import React, { useState } from 'react';
import { ShieldCheck, CheckCircle2, ChevronDown, ChevronRight, Terminal, HelpCircle, Shield } from 'lucide-react';
import { VerificationState } from '../../types/superoptimizer';

interface Stage7VerificationProps {
  verification: VerificationState;
  explainMode?: boolean;
}

export const Stage7Verification: React.FC<Stage7VerificationProps> = ({
  verification,
  explainMode = false,
}) => {
  const [showSmtLib, setShowSmtLib] = useState<boolean>(false);

  return (
    <div className="space-y-4">
      {explainMode && (
        <div className="bg-[#172231] border border-[#36D399]/40 p-3 rounded-xl text-xs text-[#E6EDF3] flex items-start space-x-2">
          <HelpCircle className="w-4 h-4 text-[#36D399] shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-[#36D399]">What is Z3 SMT Formal Verification?</span>
            <p className="text-[#8B9AAF] mt-0.5">
              Reinforcement learning policies may propose invalid rewrites. Sahelanthropus models original and optimized assembly as 32-bit BitVectors ($x_1 \dots x_{31}$) in the <strong>Z3 SMT Solver</strong>. An outcome of <strong>UNSAT</strong> guarantees that no input exists where outputs differ — mathematically proving 100% equivalence.
            </p>
          </div>
        </div>
      )}

      {/* Main Verification Status Banner */}
      <div className="bg-[#111A26] border border-[#36D399]/40 rounded-xl p-5 flex items-center justify-between shadow-xl shadow-[#36D399]/5">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 rounded-xl bg-[#36D399]/10 border border-[#36D399]/40 flex items-center justify-center">
            <ShieldCheck className="w-7 h-7 text-[#36D399]" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-lg font-bold text-[#E6EDF3] tracking-wide">
                FORMAL EQUIVALENCE VERIFICATION
              </h2>
              <span className="text-xs font-mono font-bold text-[#36D399] bg-[#36D399]/10 px-2.5 py-0.5 rounded border border-[#36D399]/40">
                STATUS: {verification.status}
              </span>
            </div>
            <p className="text-xs text-[#8B9AAF] mt-0.5">
              {verification.explanation}
            </p>
          </div>
        </div>

        <div className="text-right font-mono text-xs">
          <span className="text-[#8B9AAF] block">Z3 Latency</span>
          <span className="text-base font-bold text-[#36D399]">{verification.latencyMs} ms</span>
        </div>
      </div>

      {/* 32-Bit BitVector Symbolic Execution Register Mapping */}
      <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between border-b border-[#263445] pb-2">
          <span className="text-xs font-bold text-[#E6EDF3]">
            SYMBOLIC REGISTER EXECUTION STATE (32-BIT BITVECTORS)
          </span>
          <span className="text-[10px] font-mono text-[#00D9FF]">31 ARCHITECTURAL REGS</span>
        </div>

        <div className="overflow-x-auto bg-[#070B12] rounded-lg border border-[#263445] font-mono text-xs">
          <table className="w-full text-left">
            <thead>
              <tr className="text-[#8B9AAF] border-b border-[#263445] text-[10px]">
                <th className="p-2.5">Register</th>
                <th className="p-2.5">Original Symbolic Expression</th>
                <th className="p-2.5">Optimized Symbolic Expression</th>
                <th className="p-2.5 text-right">Verification Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#263445]/50 text-[11px]">
              {verification.symbolicRegisters.map((reg) => (
                <tr key={reg.regName} className="hover:bg-[#172231]/40">
                  <td className="p-2.5 text-[#FFB84D] font-bold">{reg.regName}</td>
                  <td className="p-2.5 text-[#E6EDF3]">{reg.origExpr}</td>
                  <td className="p-2.5 text-[#00D9FF]">{reg.optExpr}</td>
                  <td className="p-2.5 text-right">
                    <span className="text-[#36D399] font-bold flex items-center justify-end space-x-1">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>EQUIVALENT</span>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* SMT Discrepancy Query & SMT-LIB Representation */}
      <div className="bg-[#111A26] border border-[#263445] rounded-xl p-4 space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-xs font-bold text-[#E6EDF3]">
            SMT DISCREPANCY QUERY & Z3 SOLVER EXPRESSIONS
          </span>
          <button
            onClick={() => setShowSmtLib(!showSmtLib)}
            className="text-xs font-mono text-[#00D9FF] hover:underline flex items-center space-x-1"
          >
            {showSmtLib ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            <span>{showSmtLib ? 'Hide SMT-LIB Text' : 'View Generated SMT-LIB 2.0 Code'}</span>
          </button>
        </div>

        <div className="bg-[#070B12] p-3 rounded-lg border border-[#263445] font-mono text-xs space-y-1 text-[#8B9AAF]">
          <span className="text-[10px] text-[#8B9AAF]/60 block font-sans">
            SMT Existence Discrepancy Condition:
          </span>
          <code className="text-[#36D399] block">{verification.discrepancyQuery}</code>
        </div>

        {showSmtLib && (
          <div className="bg-[#070B12] p-3 rounded-lg border border-[#263445] font-mono text-xs text-[#9B7CFF] space-y-1">
            <span className="text-[10px] text-[#8B9AAF] block font-sans">Generated SMT-LIB 2.0 Script:</span>
            <pre className="text-xs leading-5 overflow-x-auto text-[#E6EDF3] font-mono">
              {verification.smtLibText}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
