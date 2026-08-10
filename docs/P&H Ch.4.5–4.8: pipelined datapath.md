1. The Pipelined Datapath (Ch. 4.5 & 4.6)
Pipelining is an implementation technique where multiple instructions are overlapped in execution. It does not reduce the time it takes to complete an individual instruction (latency) but vastly increases the number of instructions completed per unit of time (throughput).

To pipeline the RISC-V architecture, the single-cycle datapath is divided into five distinct stages, separated by Pipeline Registers that hold data and control signals between clock cycles:

IF (Instruction Fetch): Fetches the instruction from memory and increments the PC.

ID (Instruction Decode & Register Read): Decodes the instruction, generates control signals, and reads the source registers.

EX (Execute): The ALU performs operations (arithmetic, logic, or address calculation for memory).

MEM (Memory Access): Data memory is read from or written to (for load/store instructions).

WB (Write Back): The result from the ALU or Data Memory is written back to the destination register.

Pipeline Registers: The registers separating these stages are named IF/ID, ID/EX, EX/MEM, and MEM/WB. Control signals generated in the ID stage travel alongside the instruction through these registers to ensure the right signals are applied at the right time.

2. Data Hazards and RAW (Read-After-Write) (Ch. 4.7)
A Data Hazard occurs when the pipeline must be stalled because one step must wait for another to complete. The most common type is a RAW (Read-After-Write) hazard.

A RAW hazard happens when an instruction tries to read a register that a preceding, currently-executing instruction has not yet written to.

Example of a RAW Hazard:

sub x2, x1, x3  (Writes to x2 in the WB stage)

and x12, x2, x5 (Reads x2 in the ID stage)

In a standard 5-stage pipeline, the sub instruction doesn't write its result to x2 until cycle 5 (WB). However, the and instruction needs to read x2 in cycle 3 (ID). Without intervention, the and instruction would read old, incorrect data.

3. Forwarding (Bypassing) Paths
To resolve RAW hazards without severely degrading performance, processors use a technique called Forwarding (or Bypassing). Instead of waiting for the data to be written to the register file in the WB stage, the hardware grabs the data as soon as it is computed.

How Forwarding Works:
The datapath is modified by adding multiplexers at the ALU inputs. A dedicated Forwarding Unit compares the destination register (rd) of the instructions in the EX/MEM and MEM/WB pipeline registers with the source registers (rs1, rs2) of the instruction currently in the ID/EX register (about to enter the ALU).

There are two primary forwarding conditions:

EX Hazard (Forwarding from EX/MEM): The result is available directly after the ALU computes it. The EX/MEM register forwards the data back to the ALU input for the very next instruction.

Condition: EX/MEM.rd == ID/EX.rs1 (or rs2)

MEM Hazard (Forwarding from MEM/WB): The result is being written back to the register file, but a subsequent instruction (two instructions behind the producer) needs it at the ALU. The MEM/WB register forwards the data to the ALU.

Condition: MEM/WB.rd == ID/EX.rs1 (or rs2)

4. The Load-Use Data Hazard & Stalling
While forwarding solves most RAW hazards, it cannot solve all of them. A Load-Use Data Hazard occurs when an instruction relies on the result of a Load (lw) instruction immediately preceding it.

Example:

lw x2, 20(x1) (Data from memory is available at the END of the MEM stage)

and x4, x2, x5 (Needs the data at the START of the EX stage)

Because the lw instruction does not actually retrieve the data from memory until the end of its MEM stage, it is impossible to forward the data in time for the and instruction's EX stage (which happens concurrently with the Load's MEM stage). Time travel is impossible, so the hardware must stall.

The Solution:
A Hazard Detection Unit (operating in the ID stage) detects this specific Load-Use condition. When detected, it:

Forces all control signals in the ID/EX register to 0 (creating a "bubble" or a completely inactive instruction, effectively a NOP).

Prevents the PC and the IF/ID register from updating, forcing the dependent instruction and the instruction behind it to repeat their stages for one cycle.