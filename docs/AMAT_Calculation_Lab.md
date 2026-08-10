# Lab: Average Memory Access Time (AMAT) Calculation

This document outlines the step-by-step calculation for the Average Memory Access Time (AMAT) of a 2-level cache hierarchy.

### Given Parameters
*   **L1 Hit Time ($H_1$):** 1 cycle
*   **L1 Miss Rate ($M_1$):** 5% = 0.05
*   **L2 Hit Time ($H_2$):** 10 cycles
*   **L2 Miss Rate ($M_2$):** 1% = 0.01 *(Assumed as the local miss rate of the L2 cache)*
*   **DRAM Access Time (L2 Miss Penalty, $P_2$):** 200 cycles

### The AMAT Formula
For a single-level cache, the formula is:
$$ AMAT = \text{Hit Time} + (\text{Miss Rate} \times \text{Miss Penalty}) $$

For a 2-level cache hierarchy, the L1 Miss Penalty is the time it takes to access the L2 cache, plus the penalty if it misses in L2 and has to go to DRAM. This expands the formula to:
$$ AMAT = H_1 + M_1 \times (H_2 + M_2 \times P_2) $$

### Step-by-Step Calculation

**Step 1: Calculate the L1 Miss Penalty (Time spent in L2 and DRAM)**
$$ \text{L1 Miss Penalty} = H_2 + (M_2 \times P_2) $$
$$ \text{L1 Miss Penalty} = 10 + (0.01 \times 200) $$
$$ \text{L1 Miss Penalty} = 10 + 2 = 12 \text{ cycles} $$

*This means every time the CPU misses in L1, it stalls for an average of 12 cycles waiting for L2 and DRAM.*

**Step 2: Calculate the overall AMAT**
$$ AMAT = H_1 + (M_1 \times \text{L1 Miss Penalty}) $$
$$ AMAT = 1 + (0.05 \times 12) $$
$$ AMAT = 1 + 0.6 = 1.6 \text{ cycles} $$

### Final Result
The Average Memory Access Time for this hierarchy is **1.6 cycles**. 

*Context:* Even though accessing main memory takes a massive 200 cycles, the efficiency of the L1 and L2 caches brings the average access time remarkably close to the 1-cycle L1 hit time.