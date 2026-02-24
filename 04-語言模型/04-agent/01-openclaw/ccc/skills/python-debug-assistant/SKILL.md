---
name: python-debug-assistant
description: Diagnose Python runtime errors and test failures from traceback/log snippets and code context. Use when the user asks why Python code failed, requests likely root causes, wants minimal fixes, or needs a step-by-step debug plan for scripts, notebooks, APIs, or CLI tools.
---

# Python Debug Assistant

## Quick Start

- Ask for: traceback, relevant code, expected behavior, and environment (Python version, package versions).
- Return in this order: **Root cause hypothesis → Minimal fix → Verification steps → Prevention note**.
- Prefer smallest safe patch first; avoid broad rewrites unless requested.

## Workflow

1. Classify failure type: syntax, import, type, attribute, key/index, I/O, async, dependency, test assertion.
2. Identify first actionable frame in traceback (ignore cascading frames until needed).
3. Propose top 1-3 likely root causes with confidence.
4. Provide minimal patch and exact file-level change guidance.
5. Provide quick verification commands.
6. If unresolved, request the next highest-value artifact (full traceback, failing test, env lockfile).

## Output Format

Use this structure:

1. **Most likely cause** (1-2 lines)
2. **Why** (reference traceback line/function)
3. **Minimal fix** (copy-paste patch or snippet)
4. **Verify** (commands)
5. **If still failing** (next diagnostic step)

## Heuristics

- Prefer deterministic causes over speculative ones.
- For `ModuleNotFoundError`, confirm interpreter/venv mismatch before reinstall suggestions.
- For `TypeError`/`AttributeError`, check object shape at call site first.
- For flaky tests, isolate time/network/randomness and suggest mocking or seed control.
- For performance regressions, measure before optimizing.

## Verification Commands

Use these patterns when applicable:

```bash
python -V
which python
python -m pip list | head
python -m pytest -q
python -m pytest -q path/to/test_file.py::test_case -x
```

## References

- Common failure patterns: `references/common-failures.md`
- Triage checklist: `references/triage-checklist.md`
- Quick env checks script: `scripts/env_probe.sh`
