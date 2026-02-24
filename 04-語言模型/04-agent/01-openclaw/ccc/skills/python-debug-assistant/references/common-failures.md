# Common Python Failure Patterns

## 1) ModuleNotFoundError
- Check active interpreter (`which python`) and virtualenv activation.
- Ensure package installed in same interpreter (`python -m pip show <pkg>`).
- Check import name vs package name mismatch.

## 2) AttributeError on NoneType
- Upstream function may return `None` on edge path.
- Add guard clauses and assert contracts near source.

## 3) TypeError: unsupported operand / wrong argument type
- Log or inspect runtime types at call site.
- Normalize input earlier (parsing/validation layer).

## 4) KeyError / IndexError
- Validate boundary and key existence before access.
- Prefer `.get()` with explicit fallback where semantics allow.

## 5) AssertionError in tests
- Compare expected/actual snapshots.
- Check fixture drift and ordering assumptions.

## 6) Async issues (event loop / await)
- Ensure coroutine awaited.
- Avoid nested event loops in notebooks; use proper async test markers.
