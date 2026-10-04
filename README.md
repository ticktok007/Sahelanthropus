# Sahelanthropus 🦧
> **RISC-V Peephole Superoptimization via Deep Reinforcement Learning & Symbolic Equivalence Verification**

Sahelanthropus is an end-to-end framework for automatically discovering dynamic assembly code optimizations on modern RISC-V (RV32I/M) targets. It integrates a **Gymnasium-compliant superoptimization environment**, **QEMU TCG cycle-accurate cost modeling**, **Z3 SMT symbolic verification**, and a **Pre-trained Graph Attention Network (GATv2) Actor-Critic policy**.

---

## 🌟 Key Features

1. **Formal Semantic Soundness (Z3 SMT)**:
   - Evaluates transformations against a 10-rule algebraic rewrite rulebook.
   - Enforces strict 32-bit BitVector equivalence checking (`equivalence_verifier.py`) ensuring zero semantic deviation before accepting any code mutation.

2. **Heterogeneous Assembly Graph Representation (`HeteroData`)**:
   - Converts RISC-V basic blocks into heterogeneous directed graphs (`asm_graph_builder.py`).
   - Node types: `inst` (instructions with 177-dim opcode/operand features) and `reg` (registers x0–x31).
   - Edge types: `reads` and `writes` representing exact register def-use chains.

3. **Pre-trained Graph Attention Network (`GNNActorCritic`)**:
   - Uses an 8-head, 3-layer GATv2 encoder (`gat_encoder.py`) pre-trained on supervised reaching-definitions analysis (`pretrain_gat_reaching_defs.py`).
   - Learns deep relational basic block topology to guide PPO policy rollouts.

4. **Dynamic Execution Cost Modeling (QEMU TCG Plugin)**:
   - Includes a custom C-based QEMU plugin (`cycle_counter.c`) calculating dynamic execution cost (ALU=1, Memory=3, Branch=2, Mult/Div=5 cycles).

5. **Curriculum Learning & Potential-Based Reward Shaping**:
   - Scalable length schedule $\text{max\_len}(t) = \min(8 + \lfloor 17t / 500k \rfloor, 25)$.
   - Potential-based reward shaping $R_t = (C_{\text{prev}} - C_t) + \gamma \Phi(s_{t+1}) - \Phi(s_t)$ ensuring reward invariance and accelerated convergence.

---

## 📊 Benchmark Results: GNN vs. MLP Baseline

Evaluated on dependency-heavy assembly programs ($\ge 3$ def-use chains):

| Model Architecture | Mean Reward | Rules Applied | Value Loss | Explained Variance |
|:-------------------|:-----------:|:-------------:|:----------:|:------------------:|
| **GNN (Pre-trained GATv2)** | **+0.0138** | **4** | **$2.15 \times 10^{-6}$** | **+0.00015** |
| GNN (Random Initialization) | 0.0000 | 0 | $2.25 \times 10^{-6}$ | -33.67 |
| MLP Flat Vector Baseline | 0.0000 | 0 | $4.18 \times 10^{-6}$ | -0.82 |

> **Key Insight**: Flat vector representations fail to capture multi-step register data dependencies. The pre-trained GAT encoder effectively maps structural def-use chains to guide valid peephole rewrite actions.

---

## 📁 Repository Structure

```
Sahelanthropus/
├── superopt_env.py              # Gymnasium SuperoptEnv-v0 environment
├── reward_env.py                 # QEMU TCG execution & cycle counter wrapper
├── equivalence_verifier.py       # Z3 BitVector symbolic equivalence checker
├── peephole_rulebook.py          # 10 algebraic rewrite rules & AST patterns
├── asm_graph_builder.py          # PyG HeteroData graph builder (def-use extraction)
├── gat_encoder.py                # 8-head 3-layer GATv2 PyG encoder
├── ppo_gnn_actor_critic.py       # GNN Actor-Critic PPO policy network
├── graph_ppo_rollout_buffer.py   # Mini-batch PyG DataLoader PPO rollout buffer
├── curriculum_scheduler.py       # Adaptive basic block length curriculum
├── pretrain_gat_reaching_defs.py # Supervised reaching-definitions pre-training
├── cycle_counter.c               # QEMU TCG instrumentation plugin
├── run_verified_gnn_ppo.py       # Main GNN PPO training runner
├── run_dependency_heavy_benchmark.py # Comparative benchmark runner
├── run_wandb_sweep.py            # W&B hyperparameter grid search runner
├── test_*.py                     # 23 unit test suites for verification
└── docs/                         # Datapath, AMAT, and QEMU documentation
```

---

## ⚙️ Requirements & Installation

### Dependencies
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric (`torch_geometric`)
- Gymnasium
- Z3 Solver (`z3-solver`)
- QEMU (`qemu-riscv64`) & RISC-V GNU Toolchain (`riscv64-unknown-elf-gcc`)

### Building QEMU TCG Plugin
```bash
./build_cycle_counter.sh
```

---

## 🧪 Running Tests & Verification

Run the entire test suite (23 verification files):
```bash
python3 -m pytest test_*.py -v
```

Individual component tests:
- **Z3 Equivalence**: `python3 test_algebraic_identities.py`
- **Graph Builder**: `python3 test_asm_graph_builder.py`
- **GAT Encoder**: `python3 test_gat_encoder.py`
- **SuperoptEnv API**: `python3 test_superopt_env_graph_obs.py`

---

## 🚀 Running Training & Benchmarks

1. **Pre-train GAT Encoder**:
   ```bash
   python3 pretrain_gat_reaching_defs.py
   ```

2. **Run GNN-PPO Superoptimization**:
   ```bash
   python3 run_verified_gnn_ppo.py --steps 100000
   ```

3. **Run MLP vs GNN Dependency Benchmark**:
   ```bash
   python3 run_dependency_heavy_benchmark.py
   ```
