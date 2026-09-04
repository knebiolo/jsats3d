# Session Log: Handoff Note for Future AI Agents & Team

**Date:** 2026-09-03
**Subject:** Mandatory First Steps Before New Feature / HDBSCAN / Receiver Format Development

## Instructions for the Next AI Agent

Before writing code for new features, HDBSCAN multipath filter overhauls, or new receiver ingestion drivers, you **MUST** execute the following steps first:

1. **Tag the Published Baseline (`v0.1.0`)**:
   Ensure the commit corresponding to the published paper state is tagged:
   ```bash
   git tag -a v0.1.0 -m "Baseline codebase matching original published workflows"
   git push origin v0.1.0
   ```
2. **Create `CITATION.cff`**:
   Add a standard `CITATION.cff` file in the repo root containing complete citation details for Kevin Nebiolo's published paper.
3. **Update `README.md` with Citation & Reproducibility Instructions**:
   Add a "Citation" section explaining how to cite the paper and how researchers can run `git checkout v0.1.0` to reproduce the original paper results.
4. **Follow Semantic Versioning**:
   - `v0.1.0`: Original published paper state.
   - `v1.0.0`: Current modernized modular codebase (pandas 2.0+, pytest, ruff, CI).
   - `v2.0.0`: Major breaking changes (HDBSCAN multipath refactor and new receiver technology).

## Location of Instructions
These instructions are also documented in:
- `.github/copilot-instructions.md` (loaded automatically by VS Code Copilot agents)
- `.ai_journal/long_term/long_term_programming_guide.md`
