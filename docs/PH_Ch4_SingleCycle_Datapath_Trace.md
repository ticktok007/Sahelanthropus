<!-- Filename: PH_Ch4_SingleCycle_Datapath_Trace.md -->

# Single-Cycle Datapath Trace (P&H Ch 4.1 – 4.4)

This document traces the execution of four fundamental RISC-V instruction types through a single-cycle datapath. In a single-cycle implementation, the entire instruction executes in one long clock cycle, passing through up to five distinct stages: **Instruction Fetch (IF)**, **Instruction Decode (ID)**, **Execute (EX)**, **Memory Access (MEM)**, and **Write Back (WB)**.

---

## 1. Tracing `ADD rd, rs1, rs2` (R-Type)

The `ADD` instruction reads two source registers, adds their contents, and stores the result in a destination register.

*   **Instruction Fetch (IF):** 
    *   The Program Counter (PC) supplies the instruction address to the Instruction Memory.
    *   The Instruction Memory outputs the 32-bit instruction.
    *   The PC is updated: $PC \leftarrow PC + 4$.
*   **Instruction Decode (ID):** 
    *   The instruction fields are parsed. 
    *   The Register File reads the values of registers `rs1` and `rs2`.
*   **Execute (EX):** 
    *   The ALU takes the two values from the Register File as inputs.
    *   Based on the ALU control signals (derived from the instruction's `opcode`, `funct3`, and `funct7`), the ALU performs addition.
*   **Memory Access (MEM):** 
    *   No data memory access is required (`MemRead = 0`, `MemWrite = 0`). The ALU result simply bypasses this stage.
*   **Write Back (WB):** 
    *   The ALU result is routed back to the Register File.
    *   The value is written into the destination register `rd` (`RegWrite = 1`, `MemtoReg = 0`).

---

## 2. Tracing `LW rd, offset(rs1)` (I-Type)

The Load Word (`LW`) instruction computes a memory address, reads data from that address, and writes it to a register.

*   **Instruction Fetch (IF):** 
    *   Fetch the instruction from Instruction Memory using the PC.
    *   $PC \leftarrow PC + 4$.
*   **Instruction Decode (ID):** 
    *   The Register File reads the base address from register `rs1`.
    *   The 12-bit immediate (the `offset`) is passed through the ImmGen unit, which sign-extends it to 32 bits.
*   **Execute (EX):** 
    *   The ALU receives the value of `rs1` and the sign-extended immediate.
    *   The ALU adds them together to compute the effective memory address.
*   **Memory Access (MEM):** 
    *   The computed ALU address is sent to the Data Memory.
    *   The Data Memory reads the value at that address (`MemRead = 1`).
*   **Write Back (WB):** 
    *   The data read from memory is routed to the Register File.
    *   The value is written into the destination register `rd` (`RegWrite = 1`, `MemtoReg = 1`).

---

## 3. Tracing `SW rs2, offset(rs1)` (S-Type)

The Store Word (`SW`) instruction computes a memory address and writes the contents of a register into that memory location.

*   **Instruction Fetch (IF):** 
    *   Fetch the instruction from Instruction Memory using the PC.
    *   $PC \leftarrow PC + 4$.
*   **Instruction Decode (ID):** 
    *   The Register File reads two values: the base address from `rs1` and the data to be stored from `rs2`.
    *   The 12-bit immediate (split across the instruction) is sign-extended to 32 bits by the ImmGen unit.
*   **Execute (EX):** 
    *   The ALU adds the value of `rs1` and the sign-extended immediate to compute the effective memory address.
*   **Memory Access (MEM):** 
    *   The computed ALU address is sent to the Data Memory.
    *   The data value read from `rs2` is written into the Data Memory at that address (`MemWrite = 1`).
*   **Write Back (WB):** 
    *   Nothing is written back to the Register File (`RegWrite = 0`).

---

## 4. Tracing `BEQ rs1, rs2, offset` (B-Type)

The Branch if Equal (`BEQ`) instruction compares two registers and branches to a target address if they are equal.

*   **Instruction Fetch (IF):** 
    *   Fetch the instruction from Instruction Memory using the PC.
    *   The default next PC ($PC + 4$) is calculated.
*   **Instruction Decode (ID):** 
    *   The Register File reads values from `rs1` and `rs2`.
    *   The immediate value is sign-extended and shifted left by 1 bit (because instructions are half-word aligned) to create the branch offset.
*   **Execute (EX):** 
    *   The ALU performs a subtraction on `rs1` and `rs2`. 
    *   If the result is zero, it asserts the ALU's `Zero` output signal (`Zero = 1`).
    *   Simultaneously, a separate dedicated adder computes the Branch Target Address by adding the current PC to the shifted branch offset.
*   **Memory Access (MEM):** 
    *   No data memory access occurs (`MemRead = 0`, `MemWrite = 0`).
*   **Write Back (WB):** 
    *   No register is written (`RegWrite = 0`).
*   **PC Update:**
    *   A multiplexer controls the next PC value. If `Branch = 1` (control signal for BEQ) AND `Zero = 1` (from the ALU), the PC is updated to the calculated Branch Target Address. 
    *   Otherwise, the PC is updated to $PC + 4$.