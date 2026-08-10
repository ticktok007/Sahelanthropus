# P&H Chapters 5.1–5.4: Memory Hierarchy and Caches

## 5.1 Introduction: The Principle of Locality
Memory is much slower than the CPU. To bridge this gap, modern computers use a memory hierarchy (registers $\rightarrow$ L1/L2/L3 caches $\rightarrow$ Main Memory $\rightarrow$ Disk) based on the **Principle of Locality**:
*   **Temporal Locality (Locality in Time):** If a memory location is referenced, it will likely be referenced again soon (e.g., loop variables).
*   **Spatial Locality (Locality in Space):** If a memory location is referenced, locations with nearby addresses will likely be referenced soon (e.g., arrays, sequential instructions).

## 5.2 The Basics of Caches: Direct-Mapped Cache
A cache is the first level of the memory hierarchy encountered once an address leaves the CPU. 
In a **Direct-Mapped Cache**, each memory location maps to exactly one specific location in the cache. 

*   **Mapping Formula:** 
    $$ \text{Cache Index} = (\text{Block Address}) \bmod (\text{Number of Blocks in Cache}) $$
    Since cache sizes are typically powers of 2, this modulo operation is performed simply by taking the lowest $\log_2(\text{Number of Blocks})$ bits of the block address.
*   **Anatomy of a Cache Address:** A 32-bit memory address is split into three fields to access the cache:
    1.  **Index:** Used to select the specific block (row) in the cache.
    2.  **Tag:** The upper bits of the address, stored alongside the data in the cache to verify that the block currently sitting in that index actually belongs to the requested memory address.
    3.  **Offset:** The lowest bits, used to select the specific byte/word within a multi-word cache block.
*   **Valid Bit:** A single bit added to each cache entry indicating whether it contains valid data (1) or random garbage from startup (0).

**Handling Writes:**
*   **Write-Through:** Every write to the cache is simultaneously written to main memory. (Safe, but very slow unless a **Write Buffer** is used).
*   **Write-Back:** Writes only update the cache. The modified block is only written back to main memory when it is evicted to make room for a new block. Requires a "Dirty Bit" to track if it was modified.

## 5.3 Measuring and Improving Cache Performance
Cache performance fundamentally impacts CPU execution time. The processor stalls when it cannot find what it needs (a Cache Miss).
*   **CPU Time Equation:** 
    $$ \text{CPU Time} = (\text{CPU Execution Cycles} + \text{Memory Stall Cycles}) \times \text{Clock Cycle Time} $$
*   **Miss Penalty:** The time required to fetch a block from a lower level of the memory hierarchy (Main Memory) into the cache.

## 5.4 Set-Associative Caches
Direct-mapped caches suffer from **conflict misses**: if two frequently used variables map to the exact same cache index, they will constantly evict each other, even if the rest of the cache is completely empty.

To solve this, we increase associativity:
*   **Fully Associative Cache:** A block can be placed in *any* location in the cache. There is no "Index" field. To find a block, the hardware must compare the requested Tag against *every* Tag in the cache simultaneously in hardware. Very fast, but hardware-intensive and expensive.
*   **N-Way Set-Associative Cache:** A compromise between direct-mapped and fully associative. The cache is divided into "Sets", where each Set contains $N$ blocks.
    *   **Mapping Formula:** 
        $$ \text{Set Index} = (\text{Block Address}) \bmod (\text{Number of Sets in Cache}) $$
    *   A memory block maps to a specific *Set* (using the index), but can be placed in *any* of the $N$ available blocks within that Set. 
    *   To find the data, the hardware reads the Set using the index, and then compares the requested Tag against the $N$ tags in that set simultaneously.

**Block Replacement Strategies:**
When a Set is full and a new block must be brought in, which of the $N$ blocks gets evicted?
*   **Least Recently Used (LRU):** Evict the block that has gone the longest without being accessed. Hard to implement in hardware for high associativity.
*   **Random:** Pick a block at random to evict. Easier to build and performs surprisingly close to LRU for highly associative caches.