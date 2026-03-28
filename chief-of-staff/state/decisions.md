# Decision Log

> Architectural and product decisions — recorded for institutional memory.
> Format: `cos decide "Title" --context "..." --options "..." --decision "..." --rationale "..."`

---

## [2026-03-28] Use file-based JSON cache

**Date:** 2026-03-28 at 11:48

**Context:**
Need persistence across Streamlit reruns without infrastructure overhead

**Options Considered:**
1. SQLite
2. Redis
3. File-based JSON

**Decision:**
File-based JSON in /cache directory

**Rationale:**
Zero dependencies, trivially inspectable, sufficient for agent update frequency

---
