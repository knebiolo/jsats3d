# Report contract for executor runs

**For:** any model that runs a handoff plan in this repository. **From:** the external reviewer.
**Applies to:** every plan from 2026-11-13 on. A plan may add to this list. It may not remove from it.

The reviewer sees only what is committed: markdown under `docs\reports\`, the journal under `.ai_journal\session\`, and files under `docs\`. The reviewer never sees `output\`, a terminal, a plot or a database. **If it is not in a committed file, it did not happen as far as the review goes.** This page says what must be in those files so the run can be checked and the next step advised.

---

## 1. The twelve things every report contains

### 1. Run identity

- Date, and clock time at start and at end.
- The executor's model name.
- The plan file name and the commit that holds it (`git log -1 --format="%h %s" -- <plan file>`).
- `git rev-parse --abbrev-ref HEAD`, `git log -1 --format="%h %s"`, and the whole of `git status --short` (write "clean" if empty).
- The test suite summary line, when the plan asks for it.

### 2. Machine state at start and at end

- Free disk on `C:` in GB, two decimals, at both times.
- Every database the plan names: full path, size in bytes, last write time, at both times. Bytes, not MB. One table with a start column and an end column.

### 3. Every command, exactly as run

For each command: the full command line copied from the terminal, the exit code (`$LASTEXITCODE`), and the elapsed time. Not a description of the command. Not the command from the plan. The one that ran.

### 4. Console output

- For a tool run: the whole console output in a fenced block. If it is over 200 lines, the first 60 and the last 60, with a line saying how many were cut.
- For an error: the whole traceback, never a summary.

### 5. Tool output is copied, never retyped

- A markdown file written by a tool goes to `docs\reports\` with `Copy-Item`. It is not opened in an editor and not reformatted.
- After each copy: `Get-Item` on source and copy, showing equal `Length`.
- The tool file itself: `Get-FileHash -Algorithm SHA256` of the repository file and of the copy that was run. They must be equal.

### 6. Every check as a row

| Check | Expected | Observed | Result |
|---|---|---|---|

"All checks passed" without the observed values is not accepted. A check that was not run says "not run" in the Result column, with the reason.

### 7. Every number has a unit and a source

- The unit is one of: rows, solution rows, transmissions, fixes, detections, events, metres, seconds, milliseconds, bytes. "Solutions" and "positions" are not units.
- The source is a file and a table or column: "c4_tables.md, V4b, column gained".
- No number is typed from memory. No percentage is computed by hand: if the tool did not print it, it is not in the report.

### 8. Stops and the owner's words

For each STOP: the clock time, what was shown to the owner, and the owner's reply **quoted word for word**. "User confirmed" is not accepted. If the owner was not asked, write "owner not asked" and stop the run there.

### 9. Deviations, both directions

- Everything done that the plan did not ask for, with the reason.
- Everything the plan asked for that was not done, with the reason.
- "none" only if both lists are empty.

A section called "Rules observed" is not accepted. The reviewer checks the rules from the evidence above.

### 10. Things that looked wrong

A plain list of observations: a count of zero where a number was expected, a value far outside its neighbours, a file smaller than its siblings, a warning in the console. **No explanation.** One line each, with where it was seen.

### 11. Statements about cause are labelled

Any sentence that says why something happened carries one of two labels:

- `[measured: <file, table>]` when a number in a committed file shows it.
- `[not measured]` when it is a guess.

Sentences with "indicates", "suggests", "likely", "probably", "due to" or "because" need a label. The safer choice is to leave the sentence out.

### 12. What was not done, and what is asked of the reviewer

- A list of plan steps that did not run.
- Open questions, each one answerable. "Why is X?" is weak. "Table T row R shows 0 where rows 1 to 5 show about 40. Is that expected?" is strong.

---

## 2. Code the executor writes

Most plans now give a tested tool and say "write no code". Where a plan does allow a script:

- The script is saved as a `.py` file, not run as `python -c`.
- A copy goes to `docs\reports\scripts\<date>_<name>.py` so the reviewer can read it.
- The report states, for each script: which database files it opens and that they are opened read only; which table and which rows it uses, as a SQL `WHERE` clause; and the definition of each count in one sentence.
- A definition given in the plan is used as written. If it cannot be applied, STOP and say why. Never replace it with another.

---

## 3. Plots

The reviewer cannot see plots. For each plot the plan asks to be opened, answer the plan's fixed questions with yes, no or a number. Then one free line on anything that looks wrong. Do not describe a plot as "good" or "as expected".

If the owner chooses to commit a few sample PNG files, they go under `docs\reports\figures\`. That is the owner's decision, not the executor's.

---

## 4. The journal

- One entry per task, written when the task ends, with the clock time.
- Entries stay in the order they were written. Nothing is moved or rewritten later. A correction is a new entry that names the old one.
- Each entry ends with the `Get-Item` line of the files it produced.

---

## 5. Words

Not used in any report: improve, improves, improved, degrade, significant, synergistic, better, worse, justify, validated, proven, confirms (as in "confirms the hypothesis"), critical finding, key finding.

Used instead: the numbers, with their units and sources.

Times are "local time (PDT)". Never "UTC".

---

## 6. What the reviewer does with it

1. Reads the commands and outputs to see that the plan was followed.
2. Recomputes what can be recomputed from the tables (sums, differences, shares).
3. Compares the numbers with earlier committed reports.
4. Reads the code paths in the repository that produce the numbers.
5. Writes a review under `docs\reports\REVIEW_*.md` naming what is established, what is wrong (the executor's errors and the reviewer's own), and the next plan.

A report that follows this page can be reviewed in one pass. A report that does not costs a full extra run.
