# Copilot & AI Agent Instructions for jsats3d

## CRITICAL PRIORITY: Version Tagging, Citation & Reproducibility (DO THIS FIRST)

Before proceeding with any new feature development, algorithmic overhauls (such as HDBSCAN/DBSCAN multipath filter rewrites), or receiver ingestion updates, the AI agent **MUST** verify and execute the following tasks:

1. **Tag Original Published State (`v0.1.0`)**:
   Tag the baseline commit corresponding to the published paper state:
   ```bash
   git tag -a v0.1.0 -m "Baseline codebase matching original published workflows"
   git push origin v0.1.0
   ```
2. **Create `CITATION.cff`**:
   Ensure a `CITATION.cff` file exists at the root of the repository with complete citation metadata for Kevin Nebiolo's published paper.
3. **Add Citation & Reproducibility Section to `README.md`**:
   Document how researchers should cite the published paper and how to run `git checkout v0.1.0` to reproduce original paper results.
4. **Enforce Semantic Versioning**:
   - `v0.1.0`: Original published paper baseline.
   - `v1.0.0`: Current modernized modular engine (pandas >= 2.0, pytest, ruff, CI).
   - `v2.0.0`: Next major release (HDBSCAN/DBSCAN multipath pipeline and new receiver hardware formats).

---

## Architectural Policies & Coding Agreements
- Read `.ai_journal/long_term/long_term_programming_guide.md` for full project guidelines.
- Always run `pytest tests` and `ruff check jsats3d tests` before completing tasks.
- Keep per-phase session logs updated under `.ai_journal/session/`.
