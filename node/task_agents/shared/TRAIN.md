# Asking the host to measure your work

You never run a task agent yourself. You ask the host to evaluate your current tree on train samples, the same way validation evaluates a finished node, and you read the results back as files.

## The commands

`rqgm-call submit --batch batch.json [--tag "what you changed"]` snapshots your tree and asks the host to evaluate it. `batch.json` maps each role to the sample ids it should run on, for example `{"solver": ["id1", "id2"], "reviewer": ["id3"]}`. The ids are the folder names under `experience/data/train/`. `rqgm-call submit --all` takes every role the form declares instead, and `--per-role K` limits each role to its first K train samples. `--role R` takes one role.

The first submission of an episode waits until one result has settled. Later submissions return as soon as the host has admitted them, and their jobs keep running while you read results, edit code and submit again. Each submission freezes its own snapshot, so a job that finishes late still reports on the code it was given. `--timeout-s 0` returns right after admission; a positive value bounds how long the first submission waits.

`rqgm-call wait` blocks until a job has new results, without spending model turns. `rqgm-call wait --job <job>` waits for that job to finish. While either command waits, your episode gives its CPU slot to the evaluations and its idle clock is paused.

The host admits only as many rows as it has free capacity for at that moment. Rows past that are refused with `refusal_code` `insufficient_capacity` and never run; submit them again later. Both commands print one JSON reply: the job id, the submitted commit, the job's per-role counts, any other job's new results under `updates`, and the jobs still running under `pending_jobs`.

## Where the results are

`experience/generations/<depth>/results.csv` has one row per (job, role, sample) with the columns `job, ordinal, submitted_at, turn_commit, submission_commit, evaluated_commit, evaluator_commit, provenance_error, seat_digest, tag, role, task, sample, kind, success, score, refusal_code, tokens_in, tokens_out, action, judgment_action, failure_reason, failure_detail, metrics`.

- `kind` is `MEASURED` (a judge scored it), `FAILED` (the seat ran and produced nothing usable; see `failure_reason` and `failure_detail`) or `REFUSED` (the host did not run it; see `refusal_code`).
- `submission_commit` is the snapshot that was evaluated. `seat_digest` identifies its whole tree; a tag is only a label.
- `metrics` holds each deciding judge's own fields when more than one judge decides a row.

`experience/generations/<depth>/summary.json` groups the same rows: `jobs` has one entry per submission with its counts per role, `roles` covers everything so far, `by_seat` splits each role by `seat_digest`, and `lineage` gives the same split for each ancestor depth. Every group states `n`, `measured`, `failed`, `refused`, `successes`, `rate` and `score_mean`. These are counts over adaptively chosen samples, not unbiased estimates.

Both files are written by the host. Do not edit them; write your own numbers in your commit message.

Every evaluated row also leaves its record at `experience/generations/<depth>/evaluations/<action>/`, where `action` is the row's `action` column: `outputs/` (what the seat produced), `inputs/` (what it was staged), `workspace/` (files it changed), its transcript and evidence files, `judgment/<judgment action>/` for the judge's own record, and `error.json` for a failure.

## Ancestors

Your ancestors' generation folders sit beside yours, read-only, with their own `results.csv`, `summary.json` and `node-dialogue.md`. Their submission commits are in your starting Git history, so you can compare evaluated snapshots:

```bash
git diff <older-submission-commit> <newer-submission-commit> -- task_agents/ shared/ meta/
git show <submission-commit>:task_agents/<role>/<file>
```

A submission made in this episode is committed in the host store and is not in your checkout yet. To diff your own versions, commit them locally.
