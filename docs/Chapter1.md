# Patterson and Hennessy Chapter 1 Summary

### Technology Trends
The computing landscape is defined by continuous, rapid technological shifts:
* **The Post-PC Era:** The industry has transitioned from desktop-centric computing to Personal Mobile Devices (PMDs, like smartphones and tablets) and Warehouse Scale Computers (WSCs, the foundation of Cloud computing). 
* **Moore's Law:** Historically, the transistor density on integrated circuits has doubled approximately every 18 to 24 months, leading to exponential increases in memory capacity and computational power.
* **The Eight Great Ideas:** Hardware architecture is guided by recurring principles, including designing for Moore's Law, using abstraction to simplify design, making the common case fast, and improving performance via parallelism, pipelining, and prediction. 

### Performance Metrics
Measuring computer performance requires distinguishing between different types of tasks and recognizing hardware-software interactions:
* **Response Time vs. Throughput:** Response time (or execution time) measures how long it takes to complete a single task, whereas throughput measures the total amount of work completed in a given time period.
* **The Iron Law of Performance:** CPU execution time is fundamentally determined by three factors: instruction count, average Clock Cycles Per Instruction (CPI), and the clock cycle time (or clock rate). 
* **Calculation:** The relationship is expressed as: 
$$ \text{CPU Time} = \text{Instruction Count} \times \text{CPI} \times \text{Clock Cycle Time} $$
* **Standardized Benchmarks:** Because different instructions (like integer division vs. bit shifting) take wildly different amounts of cycles, performance is often quantified using standardized workload suites, such as SPEC for general computing and TPC for transaction processing.

### The Power Wall
The "Power Wall" represents the physical limit of single-processor scaling:
* **Diminishing Returns:** For decades, uniprocessor performance scaled rapidly, but this trend hit a "brick wall" due to limits in instruction-level parallelism, memory latency, and power. 
* **Thermal Limits:** The power consumed by a processor is primarily dynamic power ($P \propto C \cdot V^2 \cdot f$). Because operating voltages could no longer be continuously lowered, increasing the clock frequency caused processors to generate unmanageable amounts of heat, creating a "power wall".
* **The Sea Change:** Hitting this thermal ceiling forced a fundamental industry shift from increasing single-core clock speeds to placing multiple processor cores on a single chip. 
* **Parallelism:** To continue improving performance, the hardware now executes multiple instructions at once across multiple cores, which requires software developers to write explicitly parallel code.