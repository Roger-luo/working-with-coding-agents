# Harness Engineering

Terminal-based lecture series on coding agents and prompt engineering.

## Project Structure

- `slides.md` — combined presentation (run with `presenterm slides.md`)
- `lectures/` — individual lecture sources organized by topic
- `skills/presenterm/` — presenterm authoring skill
- `themes/` — custom presenterm themes
- `assets/` — images and other media

## Slide Authoring

- Slides use **presenterm** markdown format (not standard markdown slides)
- Slide separator: `<!-- end_slide -->` (not `---`)
- Slide titles: setext-style headers with `===` underline
- Speaker notes: `<!-- speaker_note: text -->`
- See `skills/presenterm/SKILL.md` for full syntax reference

## Git

- Use **conventional commits**: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- Scope is optional but encouraged (e.g., `docs(slides)`, `feat(demo)`)
- Keep subject line under 72 characters

## Conventions

- Keep one message per slide
- Use short lines for terminal readability (80-120 columns)
- Use `<!-- pause -->` for progressive content reveal
- Use `<!-- column_layout: [1, 1] -->` for side-by-side comparisons
- Use `<!-- incremental_lists: true -->` for bullet-by-bullet reveal
- Always `<!-- reset_layout -->` after column sections — count must match `column_layout` count
- Speaker note values must be quoted: `<!-- speaker_note: "text here" -->`
- New slides always append before the "Thank you" slide unless a specific position is requested
- Terminal has a dark background (ghostty) — avoid images with black text on white

## ASCII Art Diagrams

- Use box-drawing characters (`┌─┐│└─┘`) for diagrams
- Connect boxes vertically with `┬` (bottom) → `│` → `┴` (top of next box) — not `▼` directly above `─`
- Keep all boxes in a vertical chain the same width
- For branching, use `┬` at the split and `┴` at each target box top

- After editing diagrams, verify with: no `▼` character should have `─` directly below it in the same code block

## Slide Sections

The presentation (`slides.md`) is organized as:
1. Preliminary (audience project prompt)
2. Language Models (theory: attention, transformers, CoT, GRPO, limitations)
3. Vibe Coding and Coding Agents (what agents are, flow, tooling)
4. Prompt Engineering (context, tokens, anti-patterns, harness engineering)
5. Tool Surface, Skills, MCP
6. Live Demo: Stabilizer Tableau Simulator (incremental build with harness principles)
7. Thank you
