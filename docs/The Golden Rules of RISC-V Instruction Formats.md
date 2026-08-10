<!-- Filename: RISCV_Instruction_Formats.md -->

# RISC-V 32-bit Instruction Formats

Here is a comprehensive breakdown of the six core RISC-V 32-bit instruction formats. 

To successfully memorize these, do not try to memorize six completely different layouts. Instead, memorize the **R-Type** as your baseline, and then memorize *why* the others deviate from it.

### The Golden Rules of RISC-V Instruction Formats
RISC-V was explicitly designed to make hardware decoding as simple as possible. Because of this, certain fields **never move**:
*   **[6:0] `opcode`:** Always the bottom 7 bits.
*   **[11:7] `rd` (Destination Register):** Always bits 11 to 7 (except in S and B types, which don't write to a register).
*   **[14:12] `funct3`:** Always bits 14 to 12.
*   **[19:15] `rs1` (Source Register 1):** Always bits 19 to 15.
*   **[24:20] `rs2` (Source Register 2):** Always bits 24 to 20.
*   **[31] Sign Bit:** The highest bit of any immediate is *always* placed in bit 31 to allow for fast sign-extension before the rest of the instruction is even decoded.

---

### 1. R-Type (Register)
Used for arithmetic and logical operations requiring two source registers (e.g., `add`, `sub`, `and`). **Memorize this as your master template.**

| Bits | [31:25] | [24:20] | [19:15] | [14:12] | [11:7] | [6:0] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Field** | `funct7` | `rs2` | `rs1` | `funct3` | `rd` | `opcode` |
| **Size** | 7 bits | 5 bits | 5 bits | 3 bits | 5 bits | 7 bits |

---

### 2. I-Type (Immediate)
Used for operations requiring one register and one immediate (e.g., `addi`, `lw`, `jalr`). 
*   **The Change:** The top 12 bits (`funct7` + `rs2`) are merged to hold a 12-bit signed immediate.

| Bits | [31:20] | [19:15] | [14:12] | [11:7] | [6:0] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Field** | `imm[11:0]` | `rs1` | `funct3` | `rd` | `opcode` |
| **Size** | 12 bits | 5 bits | 3 bits | 5 bits | 7 bits |

---

### 3. S-Type (Store)
Used for storing data to memory (e.g., `sw`, `sb`). Store instructions need two source registers (base address + data to store) and an immediate (offset), but they *don't* need a destination register.
*   **The Change:** The 12-bit immediate is split. The lower 5 bits of the immediate take the place of `rd` (since stores don't write to registers). This keeps `rs1` and `rs2` exactly where they are in R-Type.

| Bits | [31:25] | [24:20] | [19:15] | [14:12] | [11:7] | [6:0] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Field** | `imm[11:5]` | `rs2` | `rs1` | `funct3` | `imm[4:0]` | `opcode` |
| **Size** | 7 bits | 5 bits | 5 bits | 3 bits | 5 bits | 7 bits |

---

### 4. B-Type (Branch)
Used for conditional branches (e.g., `beq`, `blt`). Similar to S-Type, branches need two source registers to compare, and an immediate for the branch target address.
*   **The Change:** It is almost identical to the S-Type, but the immediate bits are scrambled. RISC-V instructions must be aligned to 16-bit (half-word) boundaries, meaning the lowest bit of the immediate (`imm[0]`) is always 0. The hardware drops `imm[0]` and uses the available space to fit `imm[12]`, giving branches a wider range.

| Bits | [31] | [30:25] | [24:20] | [19:15] | [14:12] | [11:8] | [7] | [6:0] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Field** | `imm[12]` | `imm[10:5]` | `rs2` | `rs1` | `funct3` | `imm[4:1]` | `imm[11]` | `opcode` |

---

### 5. U-Type (Upper Immediate)
Used to load a 20-bit immediate into the upper 20 bits of a register (e.g., `lui`, `auipc`).
*   **The Change:** Needs a massive immediate. It drops `rs1`, `rs2`, `funct3`, and `funct7` to allocate 20 bits for the immediate, leaving only `rd` and `opcode`.

| Bits | [31:12] | [11:7] | [6:0] |
| :--- | :--- | :--- | :--- |
| **Field** | `imm[31:12]` | `rd` | `opcode` |
| **Size** | 20 bits | 5 bits | 7 bits |

---

### 6. J-Type (Jump)
Used for unconditional jumps (e.g., `jal`). It needs a large immediate for the jump target and a destination register to save the return address.
*   **The Change:** Like the U-Type, it uses the top 20 bits for an immediate. Like the B-Type, the immediate is scrambled to drop `imm[0]` (always 0) and keep the sign bit (`imm[20]`) aligned at bit 31.

| Bits | [31] | [30:21] | [20] | [19:12] | [11:7] | [6:0] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Field** | `imm[20]` | `imm[10:1]` | `imm[11]` | `imm[19:12]` | `rd` | `opcode` |