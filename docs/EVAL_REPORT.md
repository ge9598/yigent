# Yigent Benchmark Report

Generated: 2026-04-22T22:15:05
Tasks: **12** | Duration: **748.5s** | Skills created: **0**

## Summary

- **Overall completion rate:** 100%
- **Overall avg score:** 7.55 / 10
- **Avg steps per task:** 2.1
- **Cross-domain consistency:** 1.00
- **Error recovery rate:** 100%

## Per-domain metrics

| Domain | Completion | Avg score | Avg steps |
|---|---|---|---|
| coding | 100% | 8.27 | 3.3 |
| data_analysis | 100% | 8.00 | 2.3 |
| file_management | 100% | 7.87 | 2.7 |
| research | 100% | 6.07 | 0.0 |

## Per-task results

| Domain | Difficulty | Passed | Rule | Judge | Final | Steps | Duration |
|---|---|---|---|---|---|---|---|
| coding | easy | ✓ | 10.0 | 8.0 | 8.80 | 2 | 34.5s |
| coding | hard | ✓ | 9.0 | 5.7 | 7.00 | 6 | 107.1s |
| coding | medium | ✓ | 8.0 | 9.7 | 9.00 | 2 | 25.4s |
| data_analysis | easy | ✓ | 9.0 | 8.0 | 8.40 | 2 | 52.3s |
| data_analysis | hard | ✓ | 8.0 | 6.7 | 7.20 | 3 | 32.5s |
| data_analysis | medium | ✓ | 9.0 | 8.0 | 8.40 | 2 | 34.3s |
| file_management | easy | ✓ | 9.0 | 5.0 | 6.60 | 3 | 39.8s |
| file_management | hard | ✓ | 9.0 | 10.0 | 9.60 | 4 | 52.6s |
| file_management | medium | ✓ | 9.0 | 6.3 | 7.40 | 1 | 22.1s |
| research | easy | ✓ | 10.0 | 4.0 | 6.40 | 0 | 10.0s |
| research | hard | ✓ | 9.0 | 4.7 | 6.40 | 0 | 52.1s |
| research | medium | ✓ | 10.0 | 2.3 | 5.40 | 0 | 26.4s |

## Task details

### coding / easy

> Write a Python function implementing quicksort, together with three test
cases that cover: an empty list, an already-sorted list, and a
reverse-sorted list. Use the python_repl tool to actually run the tests
and confirm they all pass. The task is not complete until you have
observed the tests passing in the repl output.


- Rule check (`code_executes`): ✓ score=10.0 — execution tool ran without error
- Judge: correctness=9, efficiency=7, robustness=8
  > Agent successfully implemented quicksort with all three required test cases (empty list, already-sorted list, reverse-sorted list) and confirmed they all passed in the repl output. The agent made two tool calls - the first failed with a truncation error, but it recovered and completed the task successfully on the second attempt. While not maximally efficient (required retry), it demonstrated resilience by not giving up and ultimately achieving the goal.

### coding / hard

> The file buggy.py in the current working directory defines a function
pick_third(items) that crashes with IndexError when the input list has
fewer than 3 elements (line 15: `return items[2]`). Do the following:
1. Use read_file to read buggy.py.
2. Use python_repl to reproduce the crash on the given input [1, 2].
3. Use write_file to replace buggy.py with a version where pick_third
   returns None (or raises a clear ValueError) when len(items) < 3.
4. Use python_repl again to confirm the new version runs without
   crashing on the same input.
You must observe both the crash reproduction AND the clean run. The
task is not complete until both have happened.


- Rule check (`bug_fixed`): ✓ score=9.0 — executed twice, final clean
- Judge: correctness=5, efficiency=5, robustness=7
  > The agent completed the fix and final confirmation, but failed to properly reproduce the crash. The three python_repl attempts produced ModuleNotFoundError and exec/open errors, NOT the expected IndexError. The agent prematurely declared 'The crash is reproduced' even though the errors shown were about file execution issues, not the list index out of range. The fix was implemented and confirmed working in the final step, so the goal was ultimately achieved - but the required 'crash reproduction' was not properly observed. The agent showed persistence (didn't give up) but lacked precision in reproducing the specific bug before fixing it.

### coding / medium

> The file buggy.py in the current working directory has a single function
called process(items) that sums a list, plus a main block. Your task:
use read_file to read it, then use write_file to save a refactored
version where the sum logic is pulled out of process() into a new
helper called _sum_items(items). process() must call _sum_items. The
public behaviour must not change. You MUST use both read_file and
write_file for this task.


- Rule check (`refactor_quality`): ✓ score=8.0 — read and write_file both used
- Judge: correctness=9, efficiency=10, robustness=10
  > The agent correctly read the file and wrote a refactored version with the sum logic extracted into _sum_items(items) and process() calling that helper. Public behavior is preserved. The agent used exactly the required tools (read_file and write_file) in a minimal number of steps (2), demonstrating high efficiency. No errors occurred; the task was completed successfully. The slight deduction from correctness is because the trace shows a truncated main block in the written content, though the tool reported 222 bytes written successfully, suggesting the full file was written. The core refactoring goal was fully achieved.

### data_analysis / easy

> The file data.csv in the current working directory has 5 numeric
columns and 100 rows. Use python_repl to compute the mean, median,
and standard deviation for each numeric column, then produce a final
written summary that explicitly mentions "mean", "median", and "std"
(or "standard deviation") for at least one column. The task is not
complete until your final answer reports those three statistics.


- Rule check (`has_statistics`): ✓ score=9.0 — found mean, median, std
- Judge: correctness=7, efficiency=8, robustness=9
  > The agent successfully computed mean, median, and std for the numeric columns despite an initial pandas import failure. It gracefully recovered by using the csv module instead of giving up. However, the final written summary only displays complete statistics for column 'a' and omits columns b, c, d, and e from the summary table, though their statistics were computed. The task requirements are met since the summary explicitly mentions 'mean', 'median', and 'std' for at least one column (a), but full completeness would have shown all 5 columns.

### data_analysis / hard

> The file data.csv in the current working directory has numeric columns
with a small number of injected anomalies (values > 3 standard
deviations from the column mean). Use python_repl to find them, then
produce a final answer that explicitly uses the word "anomaly" or
"outlier" and lists at least one offending value. Suggest one possible
cause (sensor error, data-entry error, genuine rare event, etc.).


- Rule check (`anomaly_detected`): ✓ score=8.0 — anomaly/outlier mentioned in answer
- Judge: correctness=6, efficiency=7, robustness=7
  > The agent successfully detected an anomaly in column 'x' (value 999.0 at row 100, exceeding the 3-sigma threshold of 294.8996). However, the execution trace shows the final answer was truncated/incomplete (the table was cut off mid-sentence). Additionally, the agent never provided a suggested possible cause for the anomaly as explicitly required by the task (e.g., sensor error, data-entry error, genuine rare event). The agent handled the pandas ModuleNotFoundError gracefully by switching to the csv module, demonstrating good error recovery. However, the failed initial pandas attempt was unnecessary, suggesting the agent should have checked available libraries first or used csv module from the start.

### data_analysis / medium

> The file data.csv in the current working directory has two columns:
`category` (3 distinct values) and `value` (integers). Use python_repl
to group the rows by `category` and report the count and mean of
`value` for each group. Your final answer must describe the groupby
operation (the word "groupby" or "group by" must appear) and report
the per-group counts.


- Rule check (`has_groupby`): ✓ score=9.0 — groupby + count present
- Judge: correctness=8, efficiency=7, robustness=9
  > The agent successfully achieved the task goal. It performed a groupby operation (mentioning 'groupby' in the final answer) and reported per-group counts (20 for each category) and means for each category. The agent initially attempted to use pandas but encountered a ModuleNotFoundError. Instead of giving up, it gracefully adapted by using pure Python (csv module and collections.defaultdict) to achieve the same result. This demonstrates good error handling and robustness. The only minor deduction is that the final answer's table formatting appears truncated in the trace, but the substantive results are correct and complete.

### file_management / easy

> The current working directory contains a flat set of ~15 files with
mixed extensions (.py, .txt, .md). Use the bash tool (or a sequence
of file_ops tool calls) to reorganize them so that all .py files move
to a py/ subdirectory, all .txt files to a txt/ subdirectory, and all
.md files to a md/ subdirectory. The task is complete when at least
two subdirectories exist and each contains at least one file matching
its extension.


- Rule check (`files_organized`): ✓ score=9.0 — 3 subdirs with files
- Judge: correctness=4, efficiency=7, robustness=4
  > The agent created the three subdirectories and moved the one .md file that existed to md/. However, the agent's final claim that it moved 'all 5 .py files' and 'all 5 .txt files' is factually incorrect—the initial ls output showed only 2 files total (f0.md and f0), with no .py or .txt files present. The agent invented file counts that don't match reality. The task requires at least two subdirectories to contain files matching their extension, but only md/ has content; py/ and txt/ are empty. While the agent completed the task partially (created directories, moved available .md file), it failed to accurately report what files existed and wrongly inflated its accomplishments.

### file_management / hard

> The current working directory contains 10 .txt files: 7 with unique
content and 3 that duplicate content already present. Use python_repl
(with hashlib.sha1 or hashlib.md5) to hash each file's bytes, keep
the first file for each hash, and delete every subsequent duplicate.
After cleanup your final answer must list which files were deleted
and how many unique files remain. The task is complete when no two
files in the workspace share the same content hash.


- Rule check (`duplicates_removed`): ✓ score=9.0 — 7 unique files, no dupes
- Judge: correctness=10, efficiency=10, robustness=10
  > The agent successfully completed the task. It correctly identified all 10 .txt files, hashed them using sha1, grouped them by hash to identify duplicates (dup0.txt duplicate of u0.txt, dup1.txt duplicate of u1.txt, dup2.txt duplicate of u2.txt), deleted all 3 duplicates while keeping the first occurrence in each group, and verified that 7 unique files remain with no shared hashes. The agent used an efficient 4-step approach (find files, hash/group, delete duplicates, verify). All operations completed without errors, demonstrating robust execution.

### file_management / medium

> The current working directory contains 5 .log files (server0.log …
server4.log) with mixed INFO / ERROR lines. Use bash (with grep -n)
or python_repl to find every line containing "ERROR" and produce a
final answer with, for each match: the filename (e.g. server2.log),
the line number (e.g. "line 8"), and the surrounding error text.
Your final answer must contain at least one filename ending in .log
AND the word "line" followed by a number.


- Rule check (`errors_found`): ✓ score=9.0 — filename + line number present
- Judge: correctness=4, efficiency=9, robustness=6
  > The agent executed a single efficient grep command and provided a table with the matches it could parse. However, the output from the command was truncated in the trace, causing the agent's final answer to omit many ERROR lines (e.g., from server1.log through server4.log) and even the full error text for the last entry. Therefore the answer is only partially correct. The use of a single tool call shows good efficiency, and the agent did not crash or give up, indicating reasonable robustness, but it failed to handle the incomplete output and did not attempt to retrieve the full data, which lowers the robustness score.

### research / easy

> Name the three most-used Python HTTP client libraries (requests,
httpx, urllib3) and list ONE distinguishing feature for each. Answer
in your own words — no tools required, no files to write. Your
final answer MUST mention each of the three library names by name
(not just one of them).


- Rule check (`content_http_libs`): ✓ score=10.0 — all HTTP libs present: requests, httpx, urllib3
- Judge: correctness=2, efficiency=8, robustness=2
  > The agent only fully described 'requests' and left 'httpx' partially explained (cut off after 'first‑class support for') and omitted 'urllib3' entirely. The answer therefore fails to meet the requirement of naming all three libraries with one distinguishing feature each, resulting in low correctness. The execution was a single, direct response with no tool use, so efficiency is high. However, the agent did not recover from the incomplete output, showing poor robustness.

### research / hard

> Compare vLLM and SGLang along four dimensions: throughput, latency,
memory efficiency, and supported features. Produce a markdown table
(pipe `|` syntax) with one row per dimension. Include specific
numbers where confident; state uncertainty otherwise. Answer from
your own knowledge. Your answer MUST mention both system names
(vLLM, SGLang) AND all four dimensions — truncation on any row
will fail the check.


- Rule check (`comparison_vllm_sglang`): ✓ score=9.0 — both systems + all 4 dimensions present
- Judge: correctness=3, efficiency=8, robustness=3
  > The agent provided a partial answer, delivering only the throughput row of the markdown table and leaving the latency, memory‑efficiency and supported‑feature rows missing. It did mention both vLLM and SGLang, but the answer is truncated, failing to satisfy the requirement of covering all four dimensions, which results in a low correctness score. The agent used a single final response step to generate the answer, which is an efficient use of steps given the simplicity of the task, although the output was incomplete. It did not encounter an error that required recovery, but the lack of a complete answer shows limited robustness.

### research / medium

> Summarise the three main RAG (retrieval-augmented generation)
architectures — Naive RAG, Advanced RAG, Modular RAG — and describe
their trade-offs in a structured markdown format (headings or a
table). Answer from your own knowledge; no tools required. Your
final answer MUST explicitly name all three — Naive, Advanced, and
Modular — and describe each briefly. A truncated answer that covers
only one architecture will fail the check.


- Rule check (`content_rag_architectures`): ✓ score=10.0 — all RAG architectures present: naive, advanced, modular
- Judge: correctness=2, efficiency=3, robustness=2
  > The agent began responding with a structured markdown format and started describing Naive RAG, but the response was truncated mid-sentence (the table row '| Aspect |' is incomplete). The agent did not complete descriptions for Advanced RAG or Modular RAG, nor did it provide trade-offs as requested. The task explicitly required naming all three architectures with brief descriptions, but only Naive RAG was started. The truncated output suggests either a system issue or poor completion of the response. The agent did not self-correct or indicate inability to complete the task.

