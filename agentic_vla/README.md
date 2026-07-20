# Agentic-VLA Prototype Package

This folder contains the historical prototype package for the Agentic-VLA workflow.

Important:

- This package is kept for method prototyping and code reuse.
- The real paper-facing LIBERO evaluation entrypoint is `scripts/run_agentic_vla_libero.py` from the repository root.
- Results reported in project logs and the paper should come from the root-level `scripts/` pipeline, not from legacy mock or demo flows in this subpackage.

# Agentic RAG-VLM for Robotic Grasp Planning

This repository contains the core implementation of an agentic, retrieval-augmented multimodal grasp planner built on VLM reasoning.

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
