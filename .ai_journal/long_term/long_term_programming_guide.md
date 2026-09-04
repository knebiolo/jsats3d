# JSATS3D Long-Term Programming Guide & Working Agreements

## Project Overview & Mission
`jsats3d` is a Python package originally created by Kevin Nebiolo / Kleinschmidt to clean, synchronize, process, and position 3D acoustic telemetry data (JSATS) for small-scale fish tracking studies (<1000 tagged individuals, <20 receivers).

The goal of this modernization effort is to elevate `jsats3d` to modern pythonic engineering standards (Python 3.10+, pandas >= 2.0, scikit-learn, pytest, GitHub Actions CI, modular package layout).

## Architecture & Code Structure Policies
1. **Submodule Architecture**: The codebase is split into domain-specific modules under `jsats3d/`:
   - `db.py`: Database schema, initialization, and SQLite helper routines.
   - `ingest.py`: Data ingestion functions (Teknologic, acoustic raw detections, study metadata).
   - `sync.py`: Beacon processing, clock drift correction, metronome synchronization.
   - `multipath.py`: Multipath detection, feature extraction, classifier training & prediction.
   - `positioning.py`: Speed of sound estimation and Deng 3D TDOA positioning solver.
   - `density.py`: Kernel density estimation and spatial density calculations.
   - `__init__.py`: Public API export to ensure backward compatibility for notebook/script users.

2. **Data Handling & Compatibility**:
   - `pd.concat` MUST be used instead of deprecated/removed `DataFrame.append`.
   - All SQL queries interacting with SQLite MUST use parameterized placeholders (`?`) rather than string interpolation or `%s` formatters to prevent syntax errors and injection vulnerabilities.
   - Vectorized numpy/pandas operations are preferred over row-by-row iteration where performance is relevant.

3. **Error Handling & Logging**:
   - Never use bare `except:` clauses. Catch specific exceptions (e.g., `sqlite3.Error`, `ValueError`, `KeyError`, `ZeroDivisionError`).
   - Use standard Python `logging` (`logger = logging.getLogger(__name__)`) rather than print statements for warning/error/debug messages.

4. **Testing & Quality Assurance**:
   - Every core algorithm (speed of sound, database creation, time-interpolation, TDOA Deng positioning) must have unit tests using `pytest`.
   - CI workflows enforce linting and test passes on all pushes and pull requests.

## Working Agreement: The Respectful Challenger
- **Plain & Factual Communication**: Speak plainly and factually about technical trade-offs, performance implications, and architectural choices.
- **Fact-Grounded Disagreement**: Disagree constructively whenever there is a factual, safety, or risk-based reason (e.g., deprecated API removal in pandas, numerical instability in solver, SQL injection vulnerabilities).
- **Clear Recommendations & Compromises**: When proposing changes, offer a clear primary recommendation, document why, and offer a practical compromise if appropriate.
- **Protect Creative Intent & Codebase**: Prioritize the user's creative domain workflows and research goals while rigorously protecting the codebase against regressions, broken dependencies, and maintainability debt.
