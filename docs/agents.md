# Plugging in your own agent

A node is a git tree. It contains two kinds of program:

- the **meta agent** under `meta/`, which edits the tree to produce a child node (an *expansion*);
- the **task agents** under `task_agents/<role>/`, which do the domain's work (*generation*) and score other roles'
  work (*judgment*).

Both are ordinary files in the tree, so the meta agent can rewrite the task agents and itself. To start from your
own agent, change the layers the root tree is built from ([domains.md](domains.md#the-genesis-tree)).

## The container side: `node/`

Every root tree carries the code under `node/`:

| path in `node/` | path in the tree | what it is |
| --- | --- | --- |
| `shared/gateway/` | `shared/gateway/` | the gateway client: `entrypoint.py` builds a `Client` from the episode environment, `client.py` speaks the OpenAI chat or responses API with retries, plus token budget, context compaction and the prompts in `prompts.json` |
| `shared/rqgm-call`, `shared/rqgm_call.py` | `shared/` | the train client (below) |
| `task_agents/shared/` | `task_agents/shared/` | the task-agent runtime (`runtime.py`), the I/O contract check (`io_contract.py`) and `TRAIN.md` |
| `mini_swe/` | `meta/` | the shipped meta agent: upstream mini-swe-agent, pinned by `harness.json`, with `entry.py` and `solve.py` |
| `tests/` | `tests/` | contract tests for the runtime and the gateway client |

## How an episode reaches a model

Every episode runs in a Docker container. A form chooses per episode, with its `tag` in `form.json`, whether it
can call a model.

- **No model.** `"tag": null`. The container runs with `--network none` and no credential. The toy domain's task
  agents work this way.
- **A model binding.** `"tag": "<binding>"`. The host mints a grant for the episode on its gateway
  (`gm/execution/gateway.py`) and starts the container on the run's internal network. The container's environment
  carries `OPENAI_BASE_URL` (the relay on that network), `OPENAI_API_KEY` (the grant, not a provider key),
  `RQGM_AGENT_MODEL`, `RQGM_AGENT_SURFACE` and `RQGM_AGENT_TAG`. `client_from_env()` in
  `shared/gateway/entrypoint.py` reads them.

The gateway applies the binding's model, effort and output ceiling to each request, meters tokens against the
grant's allotment, records every call in the episode's `dialogue.log` and renders it into the node's
`experience/generations/<depth>/node-dialogue.md`. Provider keys stay in the host process
(`gm/execution/providers.py`). With `--provider scripted` (the default) a seeded provider answers instead: its
first reply edits a file, its second ends the episode.

## Meta agent

An expansion starts in `/workspace`, a checkout of the parent node with its git history, and runs the form's
`expansion.argv`. The meta agent's job is to leave a different tree there.

The shipped meta agent is `meta/entry.py` (imo, paper and deepswe run it directly; toy's `meta/expand.py` first
rewrites two genome files, then calls it):

- The system prompt is the shared meta prompt from `shared/gateway/prompts.json`, adjusted by the node's
  `meta/instruction.txt`. If the first line of that file is `+++ append`, `+++ prepend` or `+++ replace`, the rest
  of the file is added after, added before, or replaces the shared prompt.
- The model works through a `bash` tool in the container. It has 250 steps (`STEP_LIMIT` in `meta/solve.py`).
- The episode ends when the model runs `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, or when its token allotment
  runs out.

All of this is node code, so a meta agent can change its own prompt, loop and tools for its children.

When the episode ends, the host decides what to publish (`gm/execution/expansion.py`):

1. It waits for every train job the episode submitted. If any submission has measured rows, it restores the
   submitted tree with the best mean measured score.
2. It runs an admission probe (`gm/execution/admission_probe.py`) over a copy of the tree in a fresh container. The
   declared expansion entry point must start and exit 0, the task-agent runtime must answer a probe job, and every
   `task_agents/<role>` package must import. A tree that fails is not published.
3. If the tree differs from the parent only in host-written files, the expansion is declined. Otherwise the tree is
   sealed as the child node ([nodes.md](nodes.md#turn-chains)).

## Training: one interface for training and validation

From inside an expansion, the meta agent can ask the host to run its current task agents on train samples:

```sh
rqgm-call submit --batch batch.json        # {"solver": ["t1", "t2"], "judge": ["u1"]}
rqgm-call submit --all --per-role 2        # every role, its first 2 train samples
rqgm-call wait                             # block until new results arrive
```

`rqgm-call` writes one JSON line to the Unix socket mounted at `/run/rqgm/pool.sock` and prints the reply. The
socket identifies the episode; a request cannot name another one. The full contract given to the agent is
`task_agents/shared/TRAIN.md`.

A submission runs through the same code as a validation evaluation:

```
rqgm-call submit
  -> Pool (gm/execution/train.py): admit as many rows as free CPU allows, refuse the rest
     (insufficient_capacity), commit the current tree on the expansion's turn chain
  -> Controller.issue_train_batch (gm/controller.py): reserve, in scope `train`, one generation per
     (role, sample) and one judgment per judge slot that covers the role
  -> the same Runner (gm/execution/runner.py) and Episodes.generate / Episodes.judge
     (gm/execution/episodes.py): same pinned images, same containment, same staged input,
     same task-agent contract
  -> on settlement, settlement.owed_judgments decides which reserved judgments run: the same
     declared judges and the same settle.py
  -> Pool writes results.csv and summary.json into experience/generations/<depth>/
```

One difference in timing: a search generation's judgments are reserved after the generation settles
(`Controller.judge`), while a train submission reserves its judgments together with the generation. On
settlement, reserved judgments the outcome does not owe are closed as skips. A node judge in a train job is the
judge seat of the submitted tree itself, not the run's currently promoted judge.

| | validation (search) | training |
| --- | --- | --- |
| sample set | `roles.<role>.tasks.<task>.search` | `roles.<role>.tasks.<task>.train` |
| who triggers it, who picks samples | the HGM policy decides to evaluate a node; the kernel picks the next sample in declared order | the meta agent, with `rqgm-call submit`; it names the samples |
| evaluated tree | a published node | a submission commit on the expansion's turn chain |
| who sees results | the search policy, through the ledger; nothing is written into nodes | the submitting meta agent, as files in its generation folder, committed with the child for its descendants |
| ledger scope | `search` | `train`; excluded from the policy's snapshot and from criterion transitions |
| budget | each counted evaluation costs 1 unit of `--budget` | not counted against `--budget`; tokens are metered and recorded under scope `train` |

The runner gives train jobs priority over validation jobs, and lends an expansion's CPU to its own train jobs while
the meta agent waits.

### results.csv and summary.json

`experience/generations/<depth>/results.csv` has one row per (job, role, sample). `kind` is `MEASURED`, `FAILED` or
`REFUSED`; `score`, `success`, `submission_commit`, `seat_digest`, `failure_reason` and `refusal_code` say what
happened. `summary.json` groups the same rows by job, by role, by submitted tree (`by_seat`) and by ancestor depth
(`lineage`), with counts, success rate and mean score. Each evaluated row also leaves its full record, including
the judge's, under `experience/generations/<depth>/evaluations/<action>/`. Train samples and their labels are under
`experience/data/train/`; task agents never see that folder.

## Task agents

Each role is a package, `task_agents/<role>/__init__.py`, that exposes:

```python
def forward(subject: dict) -> dict:
    # subject["files"]      {input name: bytes}, the channel the role reads
    # subject["seed"]       int
    # subject["chat"]       callable(prompt: str) -> str, or None when the episode has no model
    # subject["solve"]      callable(task: str) -> dict running meta/solve.py as a multi-turn solver, or None
    # subject["review"]     the `review` function of the node judge of this role, or None
    # subject["evidence"]   names of the evidence files the form declares
    return {"files": {"<declared output>": b"..."}, "evidence": {"<declared evidence>": b"..."}}
```

The episode `argv` is `python -m task_agents.shared.runtime`. The runtime reads the job the host wrote to
`.rqgm/dispatch.json`, checks the inputs and the answer against `.rqgm/contract.json`
(`task_agents/shared/io_contract.py`), calls `forward` and writes the declared files. A judging role writes a file
such as `verdict.txt` holding `{"score": x}`; the form's `settle.py` turns that score into the recorded one.

## Containment

Each episode is a fresh container (`gm/execution/docker.py`) created with:

- `--network none`, or the run's internal network for an episode with a model binding;
- `--cap-drop=ALL`, `--security-opt=no-new-privileges=true`, `--ipc=private`;
- the form's image by digest, with `--pull=never`;
- `--cpus` and `--memory` from the form's `cpus` and `memory_bytes`;
- no host directory mounted, except during an expansion: the train socket, read-only, at `/run/rqgm/pool.sock`.

The workspace is copied into and out of the container as a tar stream (`docker cp`, `node/shared/workspace_tar.py`).
After the container exits, the host reads the files it left. Only declared outputs and evidence become ledger
artifacts.

The run network is created with `--internal`. Its only route out is a relay container started from the `--edge`
image, which forwards to the gateway. The gateway listens only on the Docker bridge address.
