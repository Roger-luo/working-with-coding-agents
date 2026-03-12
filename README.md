# Large Language Models & Coding Agents

[![Slides](https://img.shields.io/badge/slides-available-blue)](https://rogerluo.dev/working-with-coding-agents/)

Terminal-based presentation covering LLM internals, coding agents, prompt engineering, and harness engineering — from theory to practice.

## Contents

1. **Language Models** — self-attention, transformers, chain-of-thought, GRPO, limitations
2. **Vibe Coding and Coding Agents** — what agents are, the agent loop, tooling choices
3. **Prompt Engineering** — giving context, saving tokens, anti-patterns (over-obedience, drifting, no verification)
4. **Harness Engineering** — five principles, architecture, formalism
5. **Tool Surface, Skills, MCP** — how agents interact with the world
6. **Live Demo: Stabilizer Tableau Simulator** — incremental build applying harness principles

## Prerequisites

- **Coding agent**: [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (`npm install -g @anthropic-ai/claude-code`) or [Codex](https://github.com/openai/codex) (`npm install -g @openai/codex`)
- **Terminal**: [Ghostty](https://ghostty.org) — split into two panes (slideshow + coding agent)
- **IDE**: [Zed](https://zed.dev) or VS Code open on the side
- **Optional**: voice mode — macOS Dictation, Superwhisper, or your agent's built-in voice mode

## Usage

```bash
# Install presenterm (https://mfontanini.github.io/presenterm/introduction.html)
cargo install presenterm

# Run the presentation
presenterm slides.md
```

## Skills

Managed with [ion](https://github.com/Roger-luo/Ion). Run `ion add` to install skills from `Ion.toml`.
