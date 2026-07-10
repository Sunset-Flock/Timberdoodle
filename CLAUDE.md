# Working agreements

Read this at the start of every session and follow it.

## Conventions

1. **Do not modify or delete comments when moving code around.** When relocating
   code between files/functions, carry its comments with it verbatim. Only change
   a comment if it no longer makes sense after the move (e.g. it names the old
   location, references code that no longer surrounds it, or has become factually
   wrong) — and then update it minimally to stay accurate, don't just drop it.

2. **Use full, descriptive `snake_case` variable names — never terse abbreviations.**
   Write `channel_count`, `channel_byte_size`, `src_sample_ptr`, `average` — not
   `cc`, `cbs`, `sp`, `avg`. This applies to locals, parameters, and members alike,
   including in tight inner loops (e.g. `channel`, `texel_index`, `tap_x`/`tap_y`,
   `dst_x`/`dst_y` instead of `c`, `t`, `dx`/`dy`, `x`/`y`). Plain integer loop
   counters (`i`, `mip`) are fine when they carry no other meaning, but anything
   that abbreviates a real concept must be spelled out.

3. **Comments describe the current state of the code, not the discussion that
   produced it.** Write what the code does and why it must be that way, in the
   present tense. Do not encode the conversation or process that led there — no
   "just like B", "mirrors X", "as we decided", "previously this did Y", or
   references to review/plan task IDs (e.g. "TC.1"). A comment should read as a
   standalone statement of fact, not a story. Name another symbol only when a
   reader needs it to understand *this* code, and state the relationship directly.
   Repeating a fact in more than one place is fine when each place needs it.

4. **Keep comments brief; comment the non-obvious, don't narrate the code.** Explain
   *why*, a subtle constraint, or a gotcha — the things a reader can't see from the
   code itself. Don't restate what the code plainly says: if a `switch`/`if` chain is
   self-evident, don't enumerate its cases in a comment. When something genuinely gets
   complicated, do explain it clearly — brevity means cutting redundancy, not omitting
   what's hard. Match the codebase's form: the default is a single terse line (often
   trailing the line it explains); reserve multi-line prose for genuinely complex logic
   — tricky algorithms, concurrency invariants, non-obvious heuristics (for the long
   form done well, see `geometry_optimizer.cpp` and `thread_pool.cpp`).

5. **Describe what a function does, not what it doesn't do, and don't explain the
   world outside it.** Skip "does NOT write the .tido", "the caller then does X", and
   similar framing — describe the function's own behavior. If a function has a usage
   precondition, enforce it in the function (an assert / the type system), don't
   document it as a rule the caller must remember.

6. **Handle every enum case in a `switch`; never silently fall through.** When
   switching on an enum, list all of its values explicitly rather than covering a
   subset and letting the rest hit a permissive `default`. Give the unhandled/invalid
   case (`default`, and any value that shouldn't occur) an `DBG_ASSERT_TRUE_M(false, ...)`
   (or equivalent abort) instead of quietly returning a fallback like `0`. A new enum
   value should surface as an assert, not as a silent wrong answer.
