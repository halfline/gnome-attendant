# Contributing Guide

Thanks for helping improve this project! This guide explains how to set up your environment, the workflow to follow, and the coding style we expect.

---

## Quick start (with `uv`)

```bash
# Install the requested Python (if needed)
uv python install 3.12

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies (dev tools included)
uv sync
```

You may also use the `scripts/dwarfbind` script to use uvx and avoid dependencies.

If your editor supports it, enable “trim trailing whitespace on save” and “insert final newline.”
Do not submit changes with trailing whitespace or a missing final new line.

---

## Branches, commits, and PRs

- **Branch** from `main` for each change.
- **Commits** should be atomic and well-structured. Each commit should
represent one logical step in the project’s development. The goal is not to
preserve the exact twists and turns of drafting, but to produce a commit
history that tells a clear story of progress.

- Commit messages should aid **drive-by reviewers with limited context**.
Assume the reader does not know the project well.
- Write commit messages in the tense that reflects the state of the project **just before** the commit is applied.
- Format:
  - **First line**: a concise summary of the change being made with a short prefix (`project:`, `cli:`, `debuginfo:`, etc.).
  - **First paragraph**: summarize the code being changed (not the change itself).
  - **Second paragraph**: explain the problem with the existing state of affairs.
  - **Third paragraph**: describe how the problem is solved by the commit. Use natural prose such as “This commit addresses that by …”.
  - **Optional final paragraph**: note any future plans that will build on this change.
- Try to use natural language in commit messages. Messags should follow the above structure, but should not demarcate that structure.
- The summary line should be around 60 characters long
- All other paragraphs should wrap at 68 characters
- Reserve the demonstrative determiner "this" for the commit itself. Use "that" or other options to refer to anything else.

---

## Code style (PEP 8 compliant)

We follow PEP 8 and enforce it with `ruff` (linting) and `ruff format` (formatting).

- **Run locally**:
  ```bash
  uv run ruff check .
  uv run ruff format
  ```
- **No trailing whitespace** anywhere.
- **Line length**: 80 characters (default formatter setting).
- **Imports**: group as stdlib / third-party / local; keep them sorted.
- **Type hints**: Use them and keep them accurate.

### Variable, function, and class names

- **Descriptive, not truncated**, and **contain all vowels** where English spelling naturally includes them.
  - ✅ `total_count`, `destination_url`, `timeout_seconds`, `previous_value`, `character`, `module`
  - ❌ `ttl_cnt`, `dst_url`, `tmout_secs`, `prev_val`, `i`, `ch`, `mod`
- Prefer full words over abbreviations. Use domain terms consistently. Don't say `typedef` in one place and `type alias`
in another.
- Booleans read as assertions: `is_valid`, `has_items`, `can_retry`.

### Comments and docstrings

Write comments/docstrings that **add real value**—explain *why* and *non-obvious how*, not what the code clearly already shows.

- **Module / public API**: docstrings describing purpose, inputs/outputs, side effects, error cases.
- **Functions**: summarize behavior, assumptions, invariants, and corner cases.
- **Inline comments**: use sparingly to clarify intent or tricky logic. Do not commit commented out code, or write comments that
implicitly or explicitly allude to how the code was at some point in the past.

### Tone and language

In code, docs, commit messages, PRs, and user-visible strings:

- Keep language **humble and matter-of-fact**.
- Avoid bragging and hype. Do not use overly emphatic words such as **“comprehensive,” “crucial,” “revolutionary,” “perfect,” “obvious,” “simply.”**
- Prefer neutral phrasing: “This approach reduces allocations in the hot path” over “This is the best and fastest approach.”
- Don't assume the reader recently read the CLRS. Say "topological sort" not "Kahn's algorithm", etc.

---

## Testing

- Use `pytest`.
- Add tests for new behavior and regressions.
- Keep tests deterministic; avoid real network or clock unless the test is explicitly marked and isolated.
- Run locally:
  ```bash
  uv run pytest
  ```

---

## Tooling

These tools are expected to be available via dev dependencies:

- **ruff**: lint + format
  - Lint: `uv run ruff check .`
  - Format: `uv run ruff format`
- **pytest**: testing

---

## Adding or changing dependencies

- Use `uv add <package>` or `uv remove <package>`.
- Keep runtime dependencies minimal; prefer dev dependencies for tooling.
- Explain dependency choices in the PR description if they affect footprint or licensing.

---

## Error messages and logs

- Be specific and constructive. Offer context and next steps.
- Avoid blame or absolutes.

---

## Performance and security

- Include brief notes in the PR if your change affects complexity, memory, I/O, or security posture.
- Add benchmarks or micro-tests when a change targets performance.

---

## Review checklist (what we look for)

- PEP 8 compliance; no trailing whitespace.
- Descriptive identifiers with full words and vowels (no truncations).
- Helpful docstrings and comments; no obvious restatements.
- Clear tests covering success and failure paths.
- Humble, neutral language across code and docs.
- Minimal, justified dependencies.
- Commit log that tells a logical, reviewer-friendly story.

---

## Getting help

Open a draft PR early if feedback would help. Keep discussions respectful and focused on the code and ideas.
