# Presenterm Command Reference

Complete list of HTML comment commands supported by presenterm.

## Slide Flow

| Command | Description |
|---------|-------------|
| `<!-- end_slide -->` | Marks the boundary between slides |
| `<!-- pause -->` | Pauses rendering; next keypress reveals more content |
| `<!-- skip_slide -->` | Excludes this slide from the presentation |

## Text & Layout

| Command | Description |
|---------|-------------|
| `<!-- alignment: left -->` | Left-align subsequent content |
| `<!-- alignment: center -->` | Center subsequent content |
| `<!-- alignment: right -->` | Right-align subsequent content |
| `<!-- jump_to_middle -->` | Vertically center remaining slide content |
| `<!-- new_line -->` | Insert one blank line |
| `<!-- new_lines: N -->` | Insert N blank lines |
| `<!-- font_size: N -->` | Set font size 1-7 (kitty v0.40.0+ only) |

## Columns

| Command | Description |
|---------|-------------|
| `<!-- column_layout: [1, 1] -->` | Define column proportions |
| `<!-- column: 0 -->` | Switch to column by 0-based index |
| `<!-- reset_layout -->` | Return to default single-column layout |

## Lists

| Command | Description |
|---------|-------------|
| `<!-- incremental_lists: true -->` | Reveal list items one at a time |
| `<!-- incremental_lists: false -->` | Show all list items at once |
| `<!-- list_item_newlines: N -->` | Add N lines of spacing between list items |

## Content Inclusion

| Command | Description |
|---------|-------------|
| `<!-- include: path/to/file.md -->` | Embed contents of an external markdown file |

## Metadata & Notes

| Command | Description |
|---------|-------------|
| `<!-- speaker_note: text -->` | Speaker note (not rendered in slides) |
| `<!-- no_footer -->` | Hide footer on this slide |
| `<!-- // text -->` | Non-rendered comment |
| `<!-- comment: text -->` | Alternative comment syntax |

## Code Snippet Output

| Command | Description |
|---------|-------------|
| `<!-- snippet_output: id -->` | Reference output from an executed code snippet |
