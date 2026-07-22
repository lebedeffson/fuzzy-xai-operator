# Blind manual adjudication gate

The protocol cannot be locked until two real reviewers independently complete
all 200 rows. Reviewers receive `blind_cases.jsonl`; they must not receive H10
or baseline outputs.

For each template row, enter JSON arrays in:

- `source_elements_json`, for example `["node:preprocessor"]`;
- `optimal_cuts_json`, as an array of acceptable cuts, for example
  `[["node:preprocessor"], ["edge:preprocessor->model"]]`;
- `repair_actions_json`, using canonical low-level inverse actions;
- `ambiguous`, as `true` or `false`;
- optional notes.

Save completed files as `reviewer_1.csv` and `reviewer_2.csv`. The pipeline does
not create, infer, or prefill reviewer answers. Missing or incomplete responses
block protocol lock and sealed scoring.
