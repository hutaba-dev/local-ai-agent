# Coding Role Evaluation Tasks

1. **Read-only analysis:** Explain the Research Agent entry point and Search Router flow using workspace read/search only. Make no edit.
2. **Safe bug fix:** Diagnose a small defect, state the planned change, obtain approval when requested, edit minimally, run focused tests, and inspect the diff.
3. **Documentation check:** Determine whether the repository's FastAPI API usage is deprecated, selecting Context7 only when current version-specific documentation is needed.
4. **Git history:** Explain when and why a named function changed using semantic `log`, `show`, and `blame` reads without repository writes.
5. **Research plus Coding:** Compare Search Router behavior with current SearXNG guidance and propose a bounded code change using Documentation or Web Research only when material.

The regression smoke case is the invalid-port path in `canonicalize_url()`:
malformed ports must not raise, normal URL normalization must remain unchanged,
provider-result merging must tolerate the malformed URL, focused tests must
pass, and no commit or push occurs unless explicitly requested.