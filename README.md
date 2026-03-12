# Large Language Models & Coding Agents

Terminal-based presentation covering LLM internals, coding agents, prompt engineering, and harness engineering — from theory to practice.

## Contents

1. **Language Models** — self-attention, transformers, chain-of-thought, GRPO, limitations
2. **Vibe Coding and Coding Agents** — what agents are, the agent loop, tooling choices
3. **Prompt Engineering** — giving context, saving tokens, anti-patterns (over-obedience, drifting, no verification)
4. **Harness Engineering** — five principles, architecture, formalism
5. **Tool Surface, Skills, MCP** — how agents interact with the world
6. **Live Demo: Stabilizer Tableau Simulator** — incremental build applying harness principles

## Usage

```bash
# Install presenterm
cargo install presenterm

# Run the presentation
presenterm slides.md
```

## Skills

Managed with [ion](https://github.com/anthropics/ion). Run `ion add` to install skills from `Ion.toml`.
