MAIN CONTROL UNIT
                                       -----------------
                   Instruction[6:0] -->|               |--> RegWrite
                     (Opcode)          |               |--> ALUSrc
                                       |               |--> MemRead
                                       |               |--> MemWrite
                                       |               |--> MemtoReg
                                       |               |--> ALUOp (2b)
                                       |               |--> Branch
                                       |               |--> Jump
                                       | PCSrc Logic   |--> PCSrc (Controls NextPC MUX A)
                 ALU Zero/Status Flags>| (Branch & Zero)|
                                       |   OR Jump     |
                                       -----------------

                    Instruction[31:25] (funct7) -> \
Next PC Path        Instruction[14:12] (funct3) ->  \    ALU Control Unit
(Top Section)       ALUOp[1:0] (from Main)      ->   \--> ALU Control Signals (3-4b)


             +-----+
+-----4----- | ADD | <-- Current PC
|            +-----+
|               |
|               |                          (PC + Immediate)
|               +-----------------------\   Jump/Branch Target
|                                       v
|         NextPC MUX A                +-----+
|           +-----+    (New Address)  | ADD |
|    <0> ---|     |------------------>|     |
|   (PC+4)  |     |                   +-----+
\---------->| MUX |-->[PC Register]      ^
|    <1> ---|     |      |               |
+---------- | PCSrc|      | (Address)     |  [Immediate Shift Left 1]
|           +-----+      v               |  (B and J offsets are half-word aligned)
|PCSrc Logic|       [Instruction]        |       | Imm << 1 |
|Controls   |       [   Memory  ]        |       +----------+
|MUX A      |            |               |            ^
|           |            |               |            | (Branch/Jump Imm)
|           |            v (Instruction) |            |
|           |            | 32-bits       |            |
|           |            |               |            |
|           |            |               |            |
|           |            |            Instruction     |
|           |            |             Fields         |
|           |            |               |            |
|           |            |----------------------------+-->[Immediate]
|           |            |               |            |  [ Generator ]
|           |            |               |            |       |
\-----------/            |           Instruction      |       v (32-bit Imm)
                         |              [31:0]        \------>| ImmOut |
                         |               |                    +--------+
                         v               v (Fields)                |
                       /------------------\                        |
                      /  |   |   |   |   | \                       |
                     | 31:25|24:20|19:15|14:12|11:7| 6:0|          |
                      \ funct7|rs2|rs1 |funct3|rd |Opcode/ \       |
                       \------------------/                        |
                         |     |   |          |                    |
                         |     |   \----------+----------------\   |
                         |     v (R2)         |                |   |
                         |    rs2 index       v (R1)           |   |
                         |   /-----------> rs1 index           |   |
                         |  |                   |              |   |
                         |  |           RegWrite v             |   |
                         |  |         +----------+             |   |
                         |  |         | Register |             |   |
                         |  | (WrData)|  File    |             |   |
MUX B Controls \         |  | /------>|          |             |   |
Rs2 vs Imm      \        |  | |       | OutData1 |----\        |   |
                 \       |  | | (WrIdx)OutData2 |--\   | (D1)  |   |
               PCSrc\    |  | \--- rs2<-------/    |   v       |   |
                     \   |  |                  (D2)|  /---\    |   |
             NextPCPath\ v  |                      | | ALU |<--/   |
                        +-----+                    | \-----/       |
                    <0>-|     | (Rs2 Value)------>[|]  |           |
                        |     |                    |   | (Zero)    |
 Rs2 Value Bypass ----> | MUX |-------------------/|   \---------->| Main Control
                    <1>-|     | (Imm Bypass)       v   | (ALUResult)| Status Check
                        | ALUSrc|                 Out2 |            |
                        +-----+                        | (MEM Addr) |
                       Controls                        v            |
                       MUX B                     [ Address  ]       |
                                                 [DataMemory]       |
                                                 [          ]       |
                                                 [ Read Data]       |
                                           (WrD)>[ WriteData]       |
                                                 [ MemRead  ]       |
                                                 [ MemWrite ]       |
                                                      |             |
                                            RegWrite Signal         |
MUX C Controls \                                      |             |
ALU vs MEM      \                                     v (MemOutData)|
                 \                               PCSrc|             |
MemtoReg Signal-->|                               |   |             |
                   \                              v   | (ALUOutData)|
                    \                          +-----+|             |
WrData Path          \                    <1> -|     |v             |
                      \----------------------->|     |--\           |
MUX C Bypass---------------------------------->| MUX |---| (WrtData C Bypass)
Controls MUX D                                 |     |---|          |
                                          <0> -|     |---|          |
                                               |Memto|---|          |
                                               | Reg |---|          |
                                               +-----+---|          |
                                                         |          |
 WrData Path (bottom)                                    |          |
 Controls MUX D (Jumps)                                  |          |
                                                         |          |
Jump Signal (Main Control)---------------------------->| |          |
                                                     v v          |
                                                  +-----+         |
WrData MUX D (WB)                                 |     |         |
                                             <1>--|     |---\     |
Output to RegWrData------------------------------>| MUX |--->|    |
(for JAL PC+4)                                    | Jump|--->|WrData Path
                                             <0>--|     |--->|  (Back to RegWrData)
                                                  +-----+--->|      |
PC+4 Path Bypass                                   /         |      |
                                                  /          |      |
PC+4 Adder Out Bypass ---------------------------/           \------/