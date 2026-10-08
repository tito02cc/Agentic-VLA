# CARVE Runtime Package

This package contains the model-independent boundary shared by CARVE Agentic
Harness and CARVE Optimize Runtime.

Current runtime modules include:

- typed VLM Planner decisions and an asynchronous single-flight lifecycle;
- an episode-scoped Agentic state machine with `SAFE_HOLD`;
- deployable execution-risk monitoring and joint recovery/compute routing;
- bounded physical-recovery skills and typed expiring failure memory;
- replaceable VLA adapters for PI0.5 and OpenVLA;
- admitted deployment profiles, fidelity gates, fallback policies, and
  model/reaction/task-cycle traces.
- one `CarveAgentSession` that enforces retry/recovery budgets, forwards
  admitted inference controls, performs typed primitive verification, gates
  failure-memory writes, and persists a resumable run workspace.

Use `build_profile_admitted_tool_bindings` in `agentic_vla/assembly.py` to bind
deployment-specific observation, action execution, memory, skill, verification
and safe-hold handlers around the admitted VLA. The canonical real-model
integration smoke is `scripts/smoke_carve_canonical_models.py`; it is wiring
evidence and deliberately does not claim simulator success.

## Preserved Agentic RAG-VLM Modules

The same package also retains the earlier Agentic RAG-VLM implementation for
robotic grasp planning. Its `agent/`, `core/`, `knowledge_base/`, `config/`,
and `utils/` modules remain available and are not part of the CARVE runtime
conformance boundary.

## Highlights

- Hierarchical affordance-aware retrieval (category -> affordance -> visual).
- Scene/context-aware reasoning with structured prompting.
- Agentic planning loop with reflection and retry strategies.
- Modular codebase for VLM engine, retrieval, memory, and planning.

## Repository Scope (Clean Release)

This GitHub package intentionally keeps only the core code needed to read, run, and extend the method.

Included: - `agent/`: grasp agent, ReAct-style engine, reflection, retry logic.

- `core/`: VLM interface, retrieval modules, context builder, unified pipelines.
- `knowledge_base/`: schema and vector-store wrappers.
- `config/`: runtime configuration.
- `utils/`: helper utilities.
- `scripts/build_knowledge_base.py`: build sample KB.
- `scripts/demo_grasp.py`: single/batch demo.
- `scripts/benchmark.py`: latency/throughput benchmark.

Archived out of this folder (for paper artifact hygiene): - large result dumps, simulation-only modules, video/figure generation scripts, and other experiment utilities.

## Project Structure

```text
agentic_vla/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── agent/
├── core/
├── knowledge_base/
├── utils/
├── scripts/
│   ├── build_knowledge_base.py
│   ├── demo_grasp.py
│   └── benchmark.py
└── data/
    ├── knowledge_base_meta.json
    └── test_image.png
```

## Setup

```bash
cd agentic_vla
pip install -r requirements.txt
```

## Quick Start

1) Build a sample knowledge base

```bash
python scripts/build_knowledge_base.py
```

2) Run demo

```bash
python scripts/demo_grasp.py
```

3) Run benchmark

```bash
python scripts/benchmark.py
```

## Notes

- Some scripts include hard-coded model paths; update them to your local checkpoint path before running.
- The provided data under `data/` is lightweight and intended for quick functional verification.

## Citation

If this codebase helps your research, please cite the corresponding paper.

## License

MIT
