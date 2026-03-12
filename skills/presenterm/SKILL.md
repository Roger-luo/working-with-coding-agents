---
name: presenterm
description: >
  Create, edit, and structure terminal-based slide presentations using presenterm
  markdown format. Covers slide syntax, commands, code highlighting, columns,
  images, themes, and speaker notes.
---

# Presenterm Slide Authoring

Presenterm renders markdown files as terminal slide decks. This skill covers
the slide format, all special commands, and best practices for readable
terminal presentations.

Docs: https://mfontanini.github.io/presenterm/introduction.html

## Slide Structure

- Slides are separated by `<!-- end_slide -->` (not `---`)
- Front matter is YAML between `---` fences at the top of the file
- Setext-style headers (underlined with `===`) render as styled slide titles

```markdown
---
title: My Presentation
sub_title: Optional subtitle
author: Author Name
---

Opening Slide Title
===

Content here.

<!-- end_slide -->

Second Slide
===

More content.
```

## Commands Reference

All commands are HTML comments. For full details see `references/commands.md`.

**Pauses and flow:**
- `<!-- pause -->` — pause content until keypress
- `<!-- end_slide -->` — slide boundary
- `<!-- incremental_lists: true -->` — reveal bullets one at a time
- `<!-- incremental_lists: false -->` — disable incremental reveal

**Layout:**
- `<!-- column_layout: [1, 1] -->` — define columns with proportions
- `<!-- column: 0 -->` — switch to column by index (0-based)
- `<!-- reset_layout -->` — return to single-column
- `<!-- alignment: center -->` — text alignment (left/center/right)
- `<!-- jump_to_middle -->` — vertically center remaining content

**Spacing:**
- `<!-- new_line -->` — explicit line break
- `<!-- new_lines: 3 -->` — multiple line breaks

**Content inclusion:**
- `<!-- include: path/to/file.md -->` — embed external markdown

**Metadata:**
- `<!-- speaker_note: Your note here. -->` — speaker notes (not rendered in slides)
- `<!-- no_footer -->` — hide footer on this slide
- `<!-- skip_slide -->` — exclude slide from presentation

**Comments:**
- `<!-- // your comment -->` — non-rendered comment

## Code Blocks

Presenterm highlights 70+ languages. Key options go after the language name:

```markdown
```rust +line_numbers
fn main() { }
```
```

**Options:**
- `+line_numbers` — show line numbers
- `+exec` — allow executing the snippet with a keypress
- `+exec_replace` — execute and replace block with output
- `+no_background` — remove code block background
- `{1,3,5-7}` — highlight specific lines
- `{1-3|5-7|all}` — dynamic highlighting across sub-slides (pipe-separated groups)

**External file snippets:**

````markdown
```file +line_numbers
path: src/example.rs
language: rust
start_line: 5
end_line: 20
```
````

## Images

```markdown
![](path/to/image.png)
![image:width:50%](path/to/image.png)
```

- Paths are relative to the presentation file
- Remote URLs are not supported — use local files
- Supports kitty, iterm2, sixel protocols (auto-detected)
- Images auto-scale to preserve aspect ratio

## Columns

```markdown
<!-- column_layout: [1, 2] -->

<!-- column: 0 -->

Left column (narrower).

<!-- column: 1 -->

Right column (wider).

<!-- reset_layout -->
```

Column proportions are relative (e.g., `[1, 2]` = 1/3 + 2/3). Columns combine
well with pauses — you can show one column first and reveal the other.

## Colored Text

```markdown
<span style="color: #ff0000">Red text</span>
<span style="color: red; background-color: blue">Styled</span>
```

Use `<span class="my_class">` with a theme palette for reusable styles.

## Theme Overrides in Front Matter

```yaml
---
title: My Talk
theme:
  override:
    footer:
      style: template
      left: "My Company"
      center: "**{current_slide}** / {total_slides}"
      right: ""
      height: 3
    palette:
      classes:
        highlight:
          foreground: "#ff6600"
---
```

## Speaker Notes

Add notes inline — they are invisible during presentation.

**Important:** all command values are parsed as YAML. If a value contains
colons (`:`) or other YAML-special characters, it must be wrapped in
double quotes. Always quoting speaker notes is the safest habit.

```markdown
<!-- speaker_note: "Always quote your notes to avoid YAML parsing errors." -->
```

Run two terminals for speaker notes:
- `presenterm --publish-speaker-notes slides.md` (audience view)
- `presenterm --listen-speaker-notes slides.md` (speaker view)

## Slide Transitions

Three built-in transitions (configured in YAML or config):
- `fade`
- `slide_horizontal`
- `collapse_horizontal`

## Font Size

`<!-- font_size: N -->` (values 1-7) only works on **kitty terminal** v0.40.0+.
Theme `font_size` under `slide_title` also requires kitty. Other terminals
(ghostty, iTerm2, WezTerm) silently ignore these settings.

**Workarounds for non-kitty terminals:**

- Increase the terminal's own font size before presenting (e.g. `Cmd+=` in ghostty)
- Use a dedicated terminal profile with a larger `font-size` for presentations
- Set `defaults.max_columns` in presenterm config to limit width, giving text more room

## ASCII Art Diagrams

When creating box-drawing diagrams in code blocks:

- Use `┌─┐│└─┘` for boxes, `┬┴├┤┼` for connectors
- **Never place `▼` directly above `─`** — the arrow visually floats above the box border
- Connect boxes vertically using `┬` (bottom of box) → `│` (line) → `┴` (top of next box):

```
┌───────────┐
│   Box A   │
└─────┬─────┘
┌─────┴─────┐
│   Box B   │
└─────┬─────┘
```

- For branching, use `┬` at the split point and `┴` at each target:

```
└──┬──────┬──┘
   │      │
┌──┴──┐┌──┴──┐
│Left ││Right│
└─────┘└─────┘
```

- Keep all boxes in a vertical chain the same width

## Best Practices for Terminal Slides

- One message per slide — keep slides focused
- Short lines — terminals are typically 80-120 columns wide
- Use pauses to reveal content progressively
- Use columns for side-by-side comparisons
- Prefer ASCII diagrams over images when possible for maximum compatibility
- Dark terminal backgrounds are common — avoid images with black text on white
- Add speaker notes for non-obvious transitions
- Code slides should have working examples — use `+exec` for live demos
- Keep a static fallback if a live demo might fail
- Test with `presenterm` locally — hot reload is enabled by default

## Key Bindings (default)

| Key | Action |
|-----|--------|
| Right/Down/l/j/Space/PgDn/Enter | Next slide |
| Left/Up/h/k/PgUp | Previous slide |
| gg | First slide |
| G | Last slide |
| `<n>`G | Jump to slide n |
| Ctrl+e | Execute code snippet |
| Ctrl+p | Slide index |
| ? | Help |
| Ctrl+c / q | Quit |
