# `deepswe` seats: designed behaviour and the test that pins it

Each row is a behaviour the seats in this tree were designed to have, with the test in
`test_deepswe_seats.py` that fails when it stops being true. Run from the node root with
`python -m pytest tests/seats`.

| designed behaviour | test |
|---|---|
| The solver reads one blind channel (`input`, by position), asks its episode the statement and its context as one prose task, and publishes `model.patch` beside `assessed`, the exact bytes the reviewer judges | `test_the_solver_asks_the_episode_the_statement_and_publishes_the_reviewers_envelope` |
| The solver refuses what is not a task (no grant, no harness, no bytes, no JSON, a cell missing a field) before its episode runs | `test_the_solver_refuses_what_is_not_a_task_before_any_episode_runs` |
| The reviewer makes one chat call from the problem statement and the diff alone, on the `assessed` channel the host stages | `test_the_reviewer_judges_what_the_host_stages_and_answers_in_the_declared_files` |
| Tests in the diff are the author's own and stay in what the reviewer reads, as the author's intent; no reference solution and no undeclared name reaches the reviewer | `test_the_reviewer_sees_the_authors_own_tests_in_the_diff`, `test_the_reviewer_refuses_a_staging_that_is_not_the_work` |
| A lost `model.patch` refuses on both reviewer arms before the seat runs | `test_both_reviewer_arms_refuse_the_envelope_without_the_patch_file` |
| The verdict vocabulary is Accept (1.0) and Reject (0.0); anything else, including two decisions, is `MALFORMED` (-1.0), never a mark | `test_the_reviewer_reads_every_reply_shape_the_same_way` |
| An unreadable reply is asked again with the same text, up to `MALFORMED_RETRIES` more times | `test_an_unreadable_reply_is_asked_again_with_the_same_question` |
| A self-review carries the author's own test output, and refuses a patch or nothing in its place | `test_a_self_review_carries_the_authors_test_output_and_nothing_else_in_its_place` |
| The co-located reviewer calls reach the edited entry point and the patch on disk | `test_co_located_calls_reach_the_edited_forward_and_its_physical_patch` |
| Every seat answer is what its arm declares: the files and the evidence names | `test_every_seat_answer_is_declared_by_the_arm_that_runs_it` |
| Inside an expansion, `rqgm-call samples` lists this form's roles and shows no label | `test_rqgm_call_samples_lists_this_forms_roles_and_no_label` |

## How the work is scored

Scoring runs on the host, in the form's `settle.py`, outside this tree.

- `grader` scores the solver. It takes the task's own verifier (its `tests/` folder) and the paths the task's
  reference solution changes, both from the hidden `verifier` channel, which no seat reads. It drops from the patch
  every file the task's test patch owns, and every file matching the test-infrastructure globs (`PROTECTED`) that
  the reference solution does not change. It then runs the task's `tests/test.sh` on the rest in a fresh container
  of the task's image, with no network, all capabilities dropped and the declared CPU and memory, and takes the
  verifier's `reward`. The dropped paths are recorded as evidence (`guard.json`). A verifier that leaves no reward
  is an infrastructure failure, never a zero.
- `calls` scores the reviewer: 1 when its verdict agrees with the sample's `review_label`, else 0.
- `reviewer` is the node's own scorer of the solver; its raw score is the mark it stated, and `MALFORMED` is worth 0.

The 113 tasks are split by whole repository into train (28), validation (57) and test (28). Each task gives one
solver row and two reviewer rows, a passing and a failing model patch, in its split's `_positive` and `_negative`
halves.
