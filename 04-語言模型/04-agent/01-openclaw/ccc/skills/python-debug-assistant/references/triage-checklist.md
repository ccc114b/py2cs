# Python Debug Triage Checklist

1. Reproduce reliably
   - Exact command
   - Input/sample data
   - Deterministic seed if needed

2. Capture evidence
   - Full traceback
   - Python version + dependency versions
   - Recent code changes

3. Narrow scope
   - Identify first failing function
   - Minimize to smallest failing case

4. Patch minimally
   - One fix at a time
   - Keep behavior unchanged except target bug

5. Verify and guard
   - Re-run failing test first
   - Re-run nearby tests
   - Add regression test
