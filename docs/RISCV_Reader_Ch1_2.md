# The RISC-V Reader: Chapters 1 & 2 Summary

## Chapter 1: Design Principles (Why RISC-V?)
Unlike legacy architectures (like x86 or ARM) that have accumulated decades of obsolete instructions to maintain backwards compatibility, RISC-V is a clean-slate design. Its architecture is guided by several core principles:

*   **Cost and Simplicity:** Complexity increases silicon area, power consumption, and design verification time. RISC-V aggressively minimizes complexity. For example, it does not use condition codes (flags like Carry or Overflow) for branching, as they complicate out-of-order execution designs.
*   **Modularity over Incrementalism:** Instead of adding new instructions to a monolithic ISA over time, RISC-V uses a small, mandatory base ISA (e.g., RV32I) and optional, standardized extensions (M, A, F, D, C). Hardware designers only implement the extensions they actually need.
*   **No Microarchitecture Details in the ISA:** The ISA avoids including instructions that expose the underlying hardware implementation (like pipeline sizing or specific cache structures), ensuring that RISC-V code can run on everything from a microcontroller to a supercomputer.
*   **Open Standard:** It is free and open, allowing anyone to design, manufacture, and sell RISC-V chips without paying licensing fees, accelerating academic research and commercial innovation.

## Chapter 2: ABI Register Roles
RISC-V has 32 general-purpose registers (`x0` through `x31`). While the hardware treats `x1` through `x31` identically, the **Application Binary Interface (ABI)** assigns strict roles to each register to ensure interoperability between different software modules and compilers. 

*   **`x0` (`zero`):** Hardwired to zero. Crucial for synthesizing pseudo-instructions.
*   **`x1` (`ra` - Return Address):** Holds the address to return to after a function call. Caller-saved.
*   **`x2` (`sp` - Stack Pointer):** Points to the base of the current stack frame. Callee-saved.
*   **`x3` (`gp` - Global Pointer):** Points to global data segment to allow fast memory accesses to global variables.
*   **`x4` (`tp` - Thread Pointer):** Used for thread-local storage in multi-threaded environments.
*   **`x5–x7`, `x28–x31` (`t0–t6` - Temporaries):** Used for holding temporary values during execution. If the calling function needs these values to survive across a function call, it must save them to the stack (**Caller-saved**).
*   **`x8` (`s0` or `fp` - Saved Register/Frame Pointer):** Often used to point to the start of the local stack frame, or as a general saved register.
*   **`x9`, `x18–x27` (`s1–s11` - Saved Registers):** Used for long-lived variables. If a called function wants to use these, it is responsible for backing them up to the stack and restoring them before returning (**Callee-saved**).
*   **`x10–x11` (`a0–a1` - Arguments/Return Values):** Used to pass the first two arguments to a function, and also used by the function to return values back to the caller.
*   **`x12–x17` (`a2–a7` - Arguments):** Used to pass additional arguments to a function. (Arguments beyond `a7` must be passed via the stack). Caller-saved.