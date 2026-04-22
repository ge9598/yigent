# Yigent Benchmark Report

Generated: 2026-04-22T14:20:02
Tasks: **12** | Duration: **1250.1s** | Skills created: **0**

## Summary

- **Overall completion rate:** 83%
- **Overall avg score:** 5.80 / 10
- **Avg steps per task:** 5.3
- **Cross-domain consistency:** 0.67
- **Error recovery rate:** 100%

## Per-domain metrics

| Domain | Completion | Avg score | Avg steps |
|---|---|---|---|
| coding | 100% | 6.27 | 5.0 |
| data_analysis | 100% | 7.47 | 8.3 |
| file_management | 33% | 4.20 | 8.0 |
| research | 100% | 5.27 | 0.0 |

## Per-task results

| Domain | Difficulty | Passed | Rule | Judge | Final | Steps | Duration |
|---|---|---|---|---|---|---|---|
| coding | easy | ✓ | 10.0 | 2.7 | 5.60 | 2 | 36.9s |
| coding | hard | ✓ | 9.0 | 6.7 | 7.60 | 11 | 116.8s |
| coding | medium | ✓ | 8.0 | 4.0 | 5.60 | 2 | 39.8s |
| data_analysis | easy | ✓ | 9.0 | 6.3 | 7.40 | 6 | 118.1s |
| data_analysis | hard | ✓ | 8.0 | 6.0 | 6.80 | 11 | 156.4s |
| data_analysis | medium | ✓ | 9.0 | 7.7 | 8.20 | 8 | 89.0s |
| file_management | easy | ✗ | 0.0 | 8.3 | 5.00 | 6 | 75.6s |
| file_management | hard | ✓ | 9.0 | 5.0 | 6.60 | 11 | 168.2s |
| file_management | medium | ✗ | 0.0 | 1.7 | 1.00 | 7 | 120.0s |
| research | easy | ✓ | 9.0 | 3.7 | 5.80 | 0 | 14.3s |
| research | hard | ✓ | 8.0 | 2.3 | 4.60 | 0 | 34.8s |
| research | medium | ✓ | 9.0 | 3.0 | 5.40 | 0 | 15.8s |

## Task details

### coding / easy

> Write a Python function implementing quicksort, together with three test
cases that cover: an empty list, an already-sorted list, and a
reverse-sorted list. Use the python_repl tool to actually run the tests
and confirm they all pass. The task is not complete until you have
observed the tests passing in the repl output.


- Rule check (`code_executes`): ✓ score=10.0 — execution tool ran without error
- Judge: correctness=2, efficiency=4, robustness=2
  > The agent sent code with syntax errors and never completed the quicksort implementation. The _qs function is missing critical code (the partition logic, i assignment continuation, swapping, and recursive calls). The 'PASS' results are inconsistent with the broken code provided and appear to be from either cached execution or erroneous output. The agent did not successfully implement quicksort or demonstrate working tests.

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
- Judge: correctness=7, efficiency=5, robustness=8
  > The agent successfully achieved the functional goal: reproduced the IndexError crash and verified the fix works. However, it did not use the specified python_repl tool as required - it encountered a ModuleNotFoundError with python_repl and switched to bash with python3 instead, which is not what the task specified. The agent also created buggy.py itself (via write_file) rather than finding it already present, which suggests it may have been in a different working directory context. While the agent showed good problem-solving by finding workarounds and never gave up, it didn't follow the exact tool requirements of the task. The core objective (crash + fix verification) was met, earning partial credit.

### coding / medium

> The file buggy.py in the current working directory has a single function
called process(items) that sums a list, plus a main block. Your task:
use read_file to read it, then use write_file to save a refactored
version where the sum logic is pulled out of process() into a new
helper called _sum_items(items). process() must call _sum_items. The
public behaviour must not change. You MUST use both read_file and
write_file for this task.


- Rule check (`refactor_quality`): ✓ score=8.0 — read and write_file both used
- Judge: correctness=3, efficiency=6, robustness=3
  > The agent used both required tools (read_file and write_file), but failed to properly complete the task. The read_file result showed the file contained a `pick_third` function, not the `process` function described in the task. Instead of recognizing this mismatch and either asking for clarification or properly refactoring the existing function, the agent invented a new `process` function from scratch. This means: (1) the refactoring wasn't based on the actual file content, (2) any existing logic in the file was lost, and (3) the agent assumed facts not in evidence. Correctness is low because the output doesn't reflect proper refactoring of the actual file. Efficiency is moderate because only 2 tool calls were used. Robustness is low because the agent didn't gracefully handle the discrepancy between the task description and actual file contents—it should have flagged the mismatch rather than fabricating a solution.

### data_analysis / easy

> The file data.csv in the current working directory has 5 numeric
columns and 100 rows. Use python_repl to compute the mean, median,
and standard deviation for each numeric column, then produce a final
written summary that explicitly mentions "mean", "median", and "std"
(or "standard deviation") for at least one column. The task is not
complete until your final answer reports those three statistics.


- Rule check (`has_statistics`): ✓ score=9.0 — found mean, median, std
- Judge: correctness=6, efficiency=6, robustness=7
  > The agent successfully located the data.csv file in a subdirectory and computed statistics for at least column 'a' (mean: 49.5000, median: 49.5000, std: 29.011). However, the final answer appears truncated and only shows statistics for column 'a', not all 5 numeric columns as required. The agent handled errors gracefully (pandas unavailable, file not found) by trying alternatives, but the task appears incomplete as the summary doesn't show all 5 columns' statistics. The agent demonstrated persistence and problem-solving but didn't fully complete the output required by the task.

### data_analysis / hard

> The file data.csv in the current working directory has numeric columns
with a small number of injected anomalies (values > 3 standard
deviations from the column mean). Use python_repl to find them, then
produce a final answer that explicitly uses the word "anomaly" or
"outlier" and lists at least one offending value. Suggest one possible
cause (sensor error, data-entry error, genuine rare event, etc.).


- Rule check (`anomaly_detected`): ✓ score=8.0 — anomaly/outlier mentioned in answer
- Judge: correctness=7, efficiency=5, robustness=6
  > The agent successfully located the data.csv file after multiple failed attempts (pandas unavailable, file not in expected location) and eventually performed the 3-sigma analysis on the hard dataset, identifying value 999 as an anomaly across multiple columns. The final answer included the word 'anomalies/outliers', listed offending values (999), and attempted to suggest a cause. However, correctness is reduced because the agent didn't show the complete calculation (the trace was truncated), and it analyzed all three datasets unnecessarily. Efficiency suffered from the many failed initial attempts to load data and the decision to process multiple datasets when only one was required. Robustness was moderate - the agent recovered from errors and didn't give up, but took a circuitous path to success. The task was ultimately completed but with significant wasted effort.

### data_analysis / medium

> The file data.csv in the current working directory has two columns:
`category` (3 distinct values) and `value` (integers). Use python_repl
to group the rows by `category` and report the count and mean of
`value` for each group. Your final answer must describe the groupby
operation (the word "groupby" or "group by" must appear) and report
the per-group counts.


- Rule check (`has_groupby`): ✓ score=9.0 — groupby + count present
- Judge: correctness=9, efficiency=6, robustness=8
  > The agent successfully achieved the goal: it correctly identified the correct data.csv file (in benchmark_runs/data_analysis_medium/), performed a group by operation on the category column, and reported per-group counts (all 20) and means for each of the 3 categories. The word 'group by' appears in the final answer. The main deduction in correctness is that the agent initially tried the wrong file (data_analysis_easy) before finding the correct one with the expected 'category,value' format. For efficiency, the agent took 8 tool calls total, including multiple failed attempts (wrong file path, wrong file format, pandas not available). While the errors were handled reasonably well (falling back from pandas to csv module, using find to locate the file), the process could have been more efficient by checking file headers earlier or trying multiple paths in parallel. For robustness, the agent handled errors gracefully without giving up - when pandas wasn't available it used csv module, when the initial file wasn't found it searched subdirectories, and when one file had the wrong format it tried another until finding the correct one.

### file_management / easy

> The current working directory contains a flat set of ~15 files with
mixed extensions (.py, .txt, .md). Use the bash tool (or a sequence
of file_ops tool calls) to reorganize them so that all .py files move
to a py/ subdirectory, all .txt files to a txt/ subdirectory, and all
.md files to a md/ subdirectory. The task is complete when at least
two subdirectories exist and each contains at least one file matching
its extension.


- Rule check (`files_organized`): ✗ score=0.0 — not enough subdirectories
- Judge: correctness=9, efficiency=7, robustness=9
  > The agent successfully achieved the task goal: py/ contains buggy.py and md/ contains CLAUDE.md and README.md, satisfying the requirement of at least two subdirectories with matching files. The agent lost some efficiency points by making two incorrect path assumptions (/d/play/yigent and D:/play/yigent) before discovering the correct path via pwd (/mnt/d/play/yigent). The txt/ directory was created but remains empty since no .txt files existed in the original set. The agent demonstrated good error recovery by adapting when initial cd commands failed and proceeding with the correct path. Minor deduction from perfect correctness only because the txt/ subdirectory is empty, though this is expected given the source files available.

### file_management / hard

> The current working directory contains 10 .txt files: 7 with unique
content and 3 that duplicate content already present. Use python_repl
(with hashlib.sha1 or hashlib.md5) to hash each file's bytes, keep
the first file for each hash, and delete every subsequent duplicate.
After cleanup your final answer must list which files were deleted
and how many unique files remain. The task is complete when no two
files in the workspace share the same content hash.


- Rule check (`duplicates_removed`): ✓ score=9.0 — 7 unique files, no dupes
- Judge: correctness=4, efficiency=5, robustness=6
  > The agent achieved the core goal of removing duplicate files (ended with no duplicate hashes), but there are significant issues. 1) The task stated 10 .txt files (7 unique, 3 duplicates) but the agent found 22 files and deleted 7, which is inconsistent with the stated problem. 2) The agent had an IndentationError on first attempt, found too many files initially (including .venv), encountered a YOLO classifier block, and required multiple attempts to verify the correct workspace. 3) The final answer correctly stated which files were deleted and claimed no duplicates remain, but the numbers don't align with the original task description (10 files → should leave 7, but agent ended with 15). The agent did recover from errors and completed the task, but with poor accuracy regarding the expected input conditions.

### file_management / medium

> The current working directory contains 5 .log files (server0.log …
server4.log) with mixed INFO / ERROR lines. Use bash (with grep -n)
or python_repl to find every line containing "ERROR" and produce a
final answer with, for each match: the filename (e.g. server2.log),
the line number (e.g. "line 8"), and the surrounding error text.
Your final answer must contain at least one filename ending in .log
AND the word "line" followed by a number.


- Rule check (`errors_found`): ✗ score=0.0 — missing: filename, line
- Judge: correctness=0, efficiency=2, robustness=3
  > The agent attempted multiple approaches to locate the .log files (grep, find, ls commands) but failed to find them. It never successfully executed the core task of extracting ERROR lines from the log files. The agent did not adapt to the missing files and ultimately gave up without providing any final answer. The broad `find / -name "*.log"` command was inefficient and likely failed or would take a long time. No substantive output was produced, and no structured answer with filename, line number, and error text was ever provided. The agent needs to better handle scenarios where files are not found and should iterate on alternative approaches more systematically rather than abandoning the task.
- Error: `timeout after 120s`

### research / easy

> Name the three most-used Python HTTP client libraries (e.g. requests,
httpx, urllib3) and list ONE distinguishing feature for each. Answer in
your own words — no tools required, no files to write. Your final
answer must be at least a few sentences, not one line.


- Rule check (`content_quality`): ✓ score=9.0 — substantive answer (987 chars)
- Judge: correctness=3, efficiency=5, robustness=3
  > The agent only partially fulfilled the task by listing only one library (requests) and truncating its distinguishing feature, leaving out the other two required libraries (httpx, urllib3). It completed the response in a single step, showing reasonable efficiency, but the incomplete answer means the core requirement was not met, indicating low robustness.

### research / hard

> Compare vLLM and SGLang along four dimensions: throughput, latency,
memory efficiency, and supported features. Produce a markdown table
(use pipe `|` syntax) with one row per dimension and one column per
system. Include specific numbers where you are confident; if you are
not sure of a number, state your uncertainty rather than making one
up. Answer from your own knowledge.


- Rule check (`comparison_completeness`): ✓ score=8.0 — comparison present
- Judge: correctness=2, efficiency=3, robustness=2
  > The agent produced a severely incomplete response. The table only partially filled in the throughput row, with SGLang's entry cut off mid-sentence. Three of the four required dimensions (latency, memory efficiency, supported features) are entirely missing. No specific numbers or uncertainty statements were provided. The markdown table is incomplete and non-functional. This fails to meet the core task requirements of comparing both systems across all four specified dimensions with either concrete data or explicit uncertainty.

### research / medium

> Summarise the three main RAG (retrieval-augmented generation)
architectures — Naive RAG, Advanced RAG, Modular RAG — and describe
their trade-offs in a structured markdown format (headings or a
table). Answer from your own knowledge; no tools required. Aim for a
substantive answer of at least several hundred characters covering
all three architectures.


- Rule check (`content_quality`): ✓ score=9.0 — substantive answer (3032 chars)
- Judge: correctness=2, efficiency=5, robustness=2
  > The agent's response was incomplete, only describing the Naive RAG pipeline and providing a partially filled table before cutting off. It never mentioned Advanced RAG or Modular RAG or their trade‑offs, failing to meet the requirement for a comprehensive summary of all three architectures. While the response was generated in a single step (good efficiency), the lack of completion and failure to address the full scope of the task scores low on correctness and robustness.

