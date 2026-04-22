# Yigent Benchmark Report

Generated: 2026-04-22T11:03:15
Tasks: **12** | Duration: **661.8s** | Skills created: **0**

## Summary

- **Overall completion rate:** 17%
- **Overall avg score:** 3.15 / 10
- **Avg steps per task:** 2.3
- **Cross-domain consistency:** 0.67
- **Error recovery rate:** 0%

## Per-domain metrics

| Domain | Completion | Avg score | Avg steps |
|---|---|---|---|
| coding | 0% | 1.33 | 1.3 |
| data_analysis | 67% | 5.40 | 3.7 |
| file_management | 0% | 4.93 | 4.3 |
| research | 0% | 0.93 | 0.0 |

## Per-task results

| Domain | Difficulty | Passed | Rule | Judge | Final | Steps | Duration |
|---|---|---|---|---|---|---|---|
| coding | easy | ✗ | 0.0 | 0.0 | 0.00 | 0 | 4.1s |
| coding | hard | ✗ | 2.0 | 0.0 | 0.80 | 0 | 3.5s |
| coding | medium | ✗ | 8.0 | 0.0 | 3.20 | 4 | 180.0s |
| data_analysis | easy | ✓ | 9.0 | 4.0 | 6.00 | 0 | 24.3s |
| data_analysis | hard | ✓ | 8.0 | 7.0 | 7.40 | 10 | 171.9s |
| data_analysis | medium | ✗ | 2.0 | 3.3 | 2.80 | 1 | 14.3s |
| file_management | easy | ✗ | 0.0 | 5.3 | 3.20 | 3 | 39.9s |
| file_management | hard | ✗ | 1.0 | 7.0 | 4.60 | 8 | 12.1s |
| file_management | medium | ✗ | 3.0 | 9.7 | 7.00 | 2 | 11.5s |
| research | easy | ✗ | 2.0 | 0.0 | 0.80 | 0 | 3.5s |
| research | hard | ✗ | 3.0 | 0.0 | 1.20 | 0 | 3.0s |
| research | medium | ✗ | 2.0 | 0.0 | 0.80 | 0 | 3.4s |

## Task details

### coding / easy

> Write a Python function that implements quicksort and include 3 test cases.

- Rule check (`code_executes`): ✗ score=0.0 — no execution tool invoked
- Judge: correctness=0, efficiency=0, robustness=0
  > The agent produced no code (empty execution trace), so it failed to achieve the task of implementing quicksort with three test cases. Consequently, correctness, efficiency, and robustness cannot be assessed and are scored 0.

### coding / hard

> The following Python script raises an IndexError on line 15. Debug and fix it: [provide test script]

- Rule check (`bug_fixed`): ✗ score=2.0 — did not reproduce + verify
- Judge: correctness=0, efficiency=0, robustness=0
  > aux_error: Connection error.

### coding / medium

> Read the file src/core/agent_loop.py and refactor it to extract the tool execution logic into a separate method.

- Rule check (`refactor_quality`): ✓ score=8.0 — read and write_file both used
- Judge: correctness=0, efficiency=0, robustness=0
  > aux_error: Connection error.
- Error: `timeout after 180s`

### data_analysis / easy

> Analyze the distribution of the provided CSV file. Report mean, median, std for each numeric column and generate a summary.

- Rule check (`has_statistics`): ✓ score=9.0 — found mean, median, std
- Judge: correctness=2, efficiency=4, robustness=6
  > The agent failed to achieve the core goal of analyzing a CSV file and providing statistics. While asking for clarification about the CSV file path or content could be seen as reasonable, the task description implied a CSV file should have been provided for analysis. The agent's response was incomplete - it only acknowledged the missing information but never actually analyzed anything or generated a summary. For correctness: the agent produced no statistics or analysis (2). For efficiency: the agent took only one step but didn't complete the task (4). For robustness: the agent handled the missing information by asking for clarification, which is a reasonable approach, though it could have been more proactive in attempting to locate or assume a default file (6).

### data_analysis / hard

> Find anomalies in the provided dataset. An anomaly is any value more than 3 standard deviations from the column mean. List them and suggest possible causes.

- Rule check (`anomaly_detected`): ✓ score=8.0 — anomaly/outlier mentioned in answer
- Judge: correctness=7, efficiency=6, robustness=8
  > The agent successfully located the dataset files and performed anomaly detection analysis using a fallback approach when pandas was unavailable. It correctly identified 0 anomalies in easy/medium datasets and 3 in the hard dataset. However, the execution trace was truncated mid-report, so the full list of anomalies and suggested causes were not fully visible. For correctness: achieved partial completion of the expected 'anomaly_detected' outcome with details pending. For efficiency: used multiple steps (directory listing, file searches, failed pandas attempt, re-implementation) which could be streamlined. For robustness: excellent error handling - gracefully recovered from missing pandas module by reimplementing with csv module, handled missing dataset gracefully by searching directories, and recovered from user interaction error by continuing with file discovery.

### data_analysis / medium

> Group the provided CSV by its 'category' column and report the count and mean of the 'value' column per group. Output a summary table.

- Rule check (`has_groupby`): ✗ score=2.0 — no groupby evidence
- Judge: correctness=2, efficiency=3, robustness=5
  > The agent listed the directory but did not explore any of the visible subdirectories (coding_easy, coding_hard, data_analysis_easy, etc.) to find the CSV file. Instead of attempting to locate the data, the agent simply asked the user to provide it, which shows limited initiative. The task was not completed - no groupby operation was performed and no summary table was generated. While the agent's response was polite and it correctly identified that it didn't see an attached file, it failed to leverage the directory listing to actively search for the CSV in the available folders.

### file_management / easy

> Organize all files in the test_workspace/ directory by file extension into subdirectories (e.g., py/, txt/, md/).

- Rule check (`files_organized`): ✗ score=0.0 — not enough subdirectories
- Judge: correctness=5, efficiency=6, robustness=5
  > The agent partially achieved the goal by organizing .txt files into a txt/ subdirectory, but the task explicitly asked to organize files by extension into subdirectories (plural: 'py/, txt/, md/'). There's no evidence the agent checked for or handled other file types like .py, .md, etc. The agent made assumptions about which files existed rather than discovering all extensions first. For correctness: only completed part of the task. For efficiency: used 3 commands but could have used a more general approach (e.g., loop through extensions) instead of hardcoding 'txt'. For robustness: no error handling issues, but the agent didn't verify completeness of the task.

### file_management / hard

> Find all duplicate files in test_workspace/ by content hash, preserve the first occurrence of each, and delete the rest. Report what was deleted.

- Rule check (`duplicates_removed`): ✗ score=1.0 — 3 duplicate(s) remain
- Judge: correctness=8, efficiency=6, robustness=7
  > The agent correctly identified that all 7 files contain unique content (Content A through G), so there were no duplicates to delete. The task was to find duplicates and delete them - since no duplicates existed, nothing needed to be deleted. The agent's conclusion is accurate based on the files present. However, correctness is not 10 because the agent did not actually compute content hashes (as specified in the task) or use a hash-based approach - it simply read and visually compared file contents. For efficiency, reading all 7 files individually is acceptable but not optimal; a more efficient approach would be to compute hashes (e.g., MD5/SHA256) and group files by hash value in a single pass or use a hashmap. The agent used 8 tool calls total (1 list_dir + 7 reads), which is reasonable but could be improved. For robustness, the agent handled the task without errors, but it didn't account for potential hidden files, symbolic links, or nested directories beyond the immediate txt/ folder. Overall, the agent successfully completed the task but with room for improvement in using proper hashing and being more efficient.

### file_management / medium

> Search all .log files in test_workspace/ for lines containing 'ERROR'. Produce a summary with filename, line number, and error message.

- Rule check (`errors_found`): ✗ score=3.0 — missing: line
- Judge: correctness=10, efficiency=9, robustness=10
  > The agent correctly identified that no .log files exist in test_workspace/. It used an appropriate glob pattern '**/*.log' to search recursively, and when no matches were found, it verified the directory contents with list_dir to confirm there were only .txt files. The agent then accurately reported that no matches could be found because no .log files exist. This is the correct outcome - if there are no .log files, there cannot be any ERROR entries in them. The 'errors_found' would indeed be empty. The agent did not make assumptions and confirmed the situation through verification. Efficiency is slightly reduced from perfect because listing the directory was somewhat redundant (the search tool already indicated no .log files matched), but it's still a reasonable verification step.

### research / easy

> Name the three most-used Python HTTP client libraries and list one distinguishing feature per library.

- Rule check (`content_quality`): ✗ score=2.0 — final answer too short (0 chars)
- Judge: correctness=0, efficiency=0, robustness=0
  > aux_error: Connection error.

### research / hard

> Compare vLLM and SGLang in terms of throughput, latency, memory efficiency, and supported features. Include specific numbers where possible.

- Rule check (`comparison_completeness`): ✗ score=3.0 — no comparative structure
- Judge: correctness=0, efficiency=0, robustness=0
  > The agent produced an empty execution trace, meaning it failed to complete any part of the task. No comparison of vLLM and SGLang was provided. The agent did not attempt to gather information about throughput, latency, memory efficiency, or supported features for either framework. All scores are 0 because there was no output or work done.

### research / medium

> Summarize the three main RAG architectures (Naive, Advanced, Modular) and their trade-offs in a structured format.

- Rule check (`content_quality`): ✗ score=2.0 — final answer too short (0 chars)
- Judge: correctness=0, efficiency=0, robustness=0
  > aux_error: Connection error.

