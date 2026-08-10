# Bridging Logic and Execution: How ALU and Pipeline Concepts Connect to the RISC-V Datapath

This note synthesizes the foundational concepts of Boolean logic, Arithmetic Logic Unit (ALU) construction, and pipelining, directly applying them to the physical implementation of the RISC-V architecture. 

## 1. The ALU: From Logic Gates to the RISC-V Execute Stage

In fundamental computer architecture projects (such as Nand2Tetris), the ALU is constructed from the ground up: NAND gates form AND/OR/NOT gates, which form Half-Adders and Full-Adders, which are chained into n-bit Ripple-Carry or Carry-Lookahead Adders. The ALU is essentially a vast combinatorial logic circuit controlled by a set of instruction bits that dictate which operation's output is allowed to pass through a multiplexer (MUX) to the final output.

In the RISC-V datapath, the ALU sits squarely in the **Execute (EX) stage**. It is the computational heart of the processor, but its role extends beyond simple arithmetic:

*   **Arithmetic and Logic Operations (R-Type & I-Type):** For instructions like `add`, `sub`, `and`, and `sll`, the ALU receives two 32-bit operands (either two registers, or one register and one sign-extended immediate). The `ALU Control Unit` decodes the `funct3` and `funct7` fields from the instruction to send a 4-bit control signal to the ALU, selecting the correct internal combinatorial path.
*   **Address Calculation (Load/Store):** For `lw` and `sw`, the ALU does not perform data manipulation. Instead, it acts as an adder. It takes the base memory address (from `rs1`) and adds the sign-extended immediate offset to calculate the effective address for the Data Memory in the subsequent MEM stage.
*   **Branch Evaluation (B-Type):** For branch instructions like `beq` or `blt`, the ALU performs a subtraction between `rs1` and `rs2`. It does not write the result back to a register; instead, it uses the result to set specific status flags (like the `Zero` flag). The processor's control logic uses these flags to determine whether the `PCSrc` multiplexer should update the Program Counter (PC) to the branch target or to PC+4.

## 2. Pipelining: Breaking the Single-Cycle Bottleneck

A single-cycle datapath executes an entire instruction in one clock cycle. The fundamental flaw with this design is that the clock cycle must be long enough to accommodate the slowest instruction (usually the `lw` instruction, which must traverse the Instruction Memory, Register File, ALU, Data Memory, and Register File again). 

Pipelining solves this by dividing the datapath into five discrete hardware stages (IF, ID, EX, MEM, WB), separated by **Pipeline Registers**. 

*   **Temporal Parallelism:** Instead of waiting for one instruction to finish entirely, a pipelined datapath starts a new instruction every clock cycle. The hardware is kept constantly busy. 
*   **The Role of Pipeline Registers:** The state buffers (IF/ID, ID/EX, EX/MEM, MEM/WB) are critical. They do not just hold data; they propagate the instruction's control signals down the pipeline. When an instruction is decoded in the ID stage, all signals for the EX, MEM, and WB stages are generated immediately and passed along these registers so they arrive exactly when the instruction reaches that hardware stage.

## 3. The Collision of ALU and Pipeline: Hazards and Forwarding

The physical separation of the pipeline stages creates a disconnect between when the ALU finishes a calculation and when that calculation is officially saved to the Register File. This creates Data Hazards (specifically Read-After-Write, or RAW).

Because RISC-V relies heavily on sequential register manipulation, a naive pipeline would constantly have to stall. For example:
1. `add x1, x2, x3` (Computes x1 in EX, writes to x1 in WB)
2. `sub x4, x1, x5` (Needs x1 in EX)

By the time `sub` needs `x1` at the ALU input, the `add` instruction is only in the MEM stage. The data has not been written to the Register File yet.

**The ALU Forwarding Unit:**
To solve this, the ALU's integration into the pipelined datapath must be modified. The inputs to the ALU are no longer wired directly from the ID/EX pipeline register. Instead, they are fed through large multiplexers controlled by a **Forwarding Unit**.

This unit constantly monitors the destination registers (`rd`) of the instructions currently residing in the EX/MEM and MEM/WB pipeline registers. If it detects that a previous instruction's `rd` matches the current instruction's `rs1` or `rs2` going into the ALU, it overrides the standard register file data. It physically routes (bypasses) the output wire of the EX/MEM register or the MEM/WB register directly back to the input of the ALU.

This architectural enhancement allows the pipeline to maintain its throughput of 1 instruction per clock cycle (IPC) by feeding the ALU with "future" data before it has officially updated the system's architectural state.