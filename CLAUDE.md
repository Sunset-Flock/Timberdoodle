# Working agreements

Read this at the start of every session and follow it.

## Conventions

1. **Do not modify or delete comments when moving code around.**
   When relocating code between files/functions, carry its comments with it verbatim.
   Only change a comment if it no longer makes sense after the move.
   For example when the comment has become factually wrong — and then update it minimally to stay accurate, don't just drop it.

2. **Use full, descriptive `snake_case` variable names — never terse abbreviations.**
   Write `channel_count`, `channel_byte_size`, `src_sample_ptr`, `average` — not `cc`, `cbs`, `sp`, `avg`.
   This applies to locals, parameters, and members alike, including in tight inner loops.
   For example use `channel`, `texel_index`, `tap_x`/`tap_y`, `dst_x`/`dst_y` instead of `c`, `t`, `dx`/`dy`, `x`/`y`.
   Anything that abbreviates a real concept must be spelled out.

3. **Comments describe the current state of the code, not the discussion that produced it.**
   Write what the code does and why it must be that way, in the present tense.
   Do not encode the conversation or process that led there:
    - No "just like B", "mirrors X", "as we decided", "previously this did Y".
    - Don't include references to review/plan task IDs (e.g. "TC.1").
   A comment should read as a standalone statement of fact, not a story.
   Repeating a fact in more than one place is fine when each place needs it.

4. **Keep comments brief; comment the non-obvious, don't narrate the code.** 
   Explain *why*, a subtle constraint, or a gotcha — the things a reader can't see from the code itself.
    - Don't comment above a function about what this function does, this can be read from the implementation.
    - Don't comment above a private/public field about how this field is used - the name should already be descriptive enough.
    - If you really want to describe a field/function, describe how it should be used, not how and by whom it is used currently.
    - Don't restate what the code plainly says: if a `switch`/`if` chain is self-evident, don't enumerate its cases in a comment.
    - When something genuinely gets complicated, do explain it clearly — brevity means cutting redundancy, not omitting what's hard.
    - Match the codebase's form: the default is a single terse line above the relevant code;
    - Reserve multi-line for brief description of systems (for the long form done well, see `scene.cpp` the DESCRIPTION section or NOTES sections).

5. **Handle every enum case in a `switch`; never silently fall through.**
   When switching on an enum, list all of its values explicitly rather than covering a subset and letting the rest hit a permissive `default`.
   Give the unhandled/invalid case (`default`, and any value that shouldn't occur) an `DBG_ASSERT_TRUE_M(false, ...)`
   Don't quietly return a fallback like `0`.
   A new enum value should surface as an assert, not as a silent wrong answer.
