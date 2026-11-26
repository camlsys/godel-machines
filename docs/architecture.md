# Architecture

The codebase runs two machines with one kernel. The Huxley-Gödel Machine (HGM) searches under a fixed evaluation
criterion; the Red Queen Gödel Machine (RQGM) also moves the criterion at checkpoints, as the form's `transitions`
declare (judge promotion, curricula, interpretations, adversarial pools). Both use the same search policy (`Hgm`,
below); the only difference is whether the form declares transitions (`gm/transitions.py`).

## One run

```
python -m gm run --domain toy/mini_swe ...
  gm/__main__.py           parse options, choose the provider
  gm/run.py                load_form: read form.json and data/, resolve image digests;
                           start: seed the root node and dataset tables, write run-configured;
                           run: open the runner, drive the controller, return the report
  gm/controller.py         loop until the budget is spent:
      settle what has arrived -> build the snapshot from the fold -> the policy decides:
      expand a node, or evaluate one on a cell (the kernel picks the sample)
      reserve the action (action-reserved) -> submit a job to the work queue
      take the next completion -> settle it (expansion-recorded / task-outcome / judgment-recorded / soft-skip)
      reserve the judgments a settled outcome owes; at due checkpoints, compare and move criteria
  gm/execution/            run each job as a Docker episode; model calls go through the gateway
```

The controller is the only writer to the ledger. Execution runs jobs on worker threads and returns completions;
the controller applies them in arrival order. Train submissions from inside an expansion reserve through the same
controller, under its lock.

## The policy seam

`gm/policy.py` is the whole interface between the kernel and a search policy.

- `Snapshot`: one frozen decision state, built from the fold by `Sampler.snapshot` in `gm/sampling.py`. It holds
  the archive (`parents`: each searchable node's parent, `None` for a root), each role's evaluation cells
  (`roles`), each cell's counted evidence (`evidence`, Beta pseudo-counts: `score_sum` S and `trials` S + F over
  fractional outcomes in [0, 1]), each cell's completed and reserved positions, the evaluations counted so far
  (`clock_completed`) and the `budget`, the decision index, which cells are exhausted (`closed`), and the
  execution state of open expansions. It holds nothing a run would not know.
- `Expansion(parent)` and `Evaluation(node, role, task)`: the two actions. An evaluation names a cell, never a
  sample.
- `Policy`: any object with `decide(snapshot) -> Action`. It raises `NoAction` when it has nothing to issue now;
  the controller then waits for completions.
- `snapshot.draws`: seeded randomness keyed by the run seed and the decision index (`Draws.uniform`,
  `Draws.thompson`, `Draws.argmax`), so the same ledger prefix gives the same decision.
- `evidence_maps(snapshot)`: each node's own evidence and its clade evidence (pooled over the node and its
  descendants).
- `pick(snapshot)`: the final selection, the node with the highest `BB_eps = I^-1_eps(1 + S, 1 + F)` over its own
  evidence, `eps` = 0.05. The run report carries it.
- `Hgm`: the shipped policy. Thompson draws over clade evidence choose a parent to expand and over node evidence a
  node to evaluate. The tree widens while `evaluations^alpha` reaches the published width (alpha 2/3), after a
  bootstrap cohort of `floor(workers^alpha)` root expansions. Within a node it reads the least-loaded cell.

The kernel keeps three things out of the policy's hands:

- **Sample choice.** `Sampler.next_sample` walks the cell's sample set in declared order, skipping samples an open
  action holds or that await a verdict. Every record is therefore valid for any policy.
- **Admission.** Each (kind, sample set) cell admits one open action at a time; the budget is charged one unit per
  counted evaluation, and an open evaluation holds one unit until it settles (`Allocations` in
  `gm/controller.py`).
- **Expansion questions.** A parent whose expansion came back without a child is not asked the same question
  again (same chain evidence, same pins, `gm/ledger/question.py`) until new evidence lands, beyond a bounded
  number of re-asks.

When the policy is `Hgm`, the controller also enforces its widening gates at issue time, so concurrent workers
cannot over-commit evaluations past the next expansion's need. A custom policy owns its own gates.

## Modules

| part | where | what it does |
| --- | --- | --- |
| CLI | `gm/__main__.py` | `run` and `verify` |
| run | `gm/run.py` | `forms`, `load_form`, `initial_pins`, `start`, `run(form, ..., policy=)` |
| controller | `gm/controller.py` | decide, reserve, submit, settle, judge, checkpoint; budget allocations; train batches |
| policy | `gm/policy.py` | snapshot, actions, `Policy`, keyed draws, `Hgm`, `pick` |
| sampling | `gm/sampling.py` | populations, the sample walk, per-cell caps, exhausted cells, the snapshot |
| settlement | `gm/settlement.py` | completion to event; clock units; which judgments a settled outcome owes |
| transitions | `gm/transitions.py` | checkpoint schedule, rescoring, labeled cells over minted pools, and the declared rules (curriculum, pool mint, objective promotion by BB_eps) |
| ledger | `gm/ledger/` | event codec, SQLite log, the fold and its indexes, the question digest, settings ([ledger.md](ledger.md)) |
| domain | `gm/domain/` | one form and its data read into a `DomainManifest` ([domains.md](domains.md)) |
| node store | `gm/git_store/` | nodes as git commits ([nodes.md](nodes.md)) |
| work queue | `gm/work_queue.py` | the jobs, completions and the `WorkQueue` protocol |
| runner | `gm/execution/runner.py` | CPU admission by class priority (expansion, then train, then validation), CPU lending to an expansion's own train jobs, bounded retries |
| episodes | `gm/execution/episodes.py`, `expansion.py`, `admission_probe.py` | stage a workspace, run the container, harvest outputs; publish an expansion's child |
| containment | `gm/execution/docker.py` | image digests, container argv, the run network and its relay, tar transfer |
| gateway | `gm/execution/gateway.py`, `providers.py` | grants, token metering, dialogue records; LiteLLM for live models, a seeded scripted provider offline |
| train pool | `gm/execution/train.py` | serves `rqgm-call` from inside an expansion; writes `results.csv` and `summary.json` |
| node runtime | `node/` | copied into every root tree: gateway client, `rqgm-call`, task-agent runtime, `mini_swe` |

## Determinism

- Events are canonical JSON hashed with sha256. Nodes are git commits with a fixed author, committer and date. The
  same facts get the same ids on any machine.
- Every draw is computed from the run seed and the decision index, naming nodes by publication order. The same
  ledger prefix gives the same decision.
- With `--workers 1` the controller settles each job before the next decision, and a run with the scripted provider
  is byte-reproducible from its seed: ledger and node commits. With more workers, the ledger records completions in
  the order they arrive.

## Cost per decision

A decision does not rescan the ledger. The fold (`gm/ledger/projection.py`) applies one event at a time and keeps
the indexes decisions read: per-cell evidence (`trial_index.py`), per-cell completed and awaiting samples
(`membership_index.py`), schedule positions (`schedule_index.py`), judgment streams (`judged_index.py`), and each
open action's issue-time pins. The snapshot is built once per ledger head; an exhausted-cell answer is re-derived
only when that cell's membership, holds or population change. Building the snapshot costs one pass over the
searchable nodes' cells, so it grows with node count, not with ledger length. The question digest caches each
chain node's share and rebuilds it only when that node gets new evidence or the pins move.

## Older ledgers

A ledger written by the previous event schema (8) decodes, re-encodes to its stored bytes and folds through the
same codec. Its decisions are keyed on its own recorded settings: `gm/ledger/settings.py` maps a run's manifest to
the settings the policy, sampling, caps and transition rules read (constants for a run this schema starts; the
recorded configuration, draw keys and sampling passes for a schema-8 run).

## Replacing the executor

The controller talks to execution only through `WorkQueue` (`gm/work_queue.py`): `submit`, `next_completion` and
`expansion_execution`. To use another executor, such as a cluster scheduler or a different container runtime,
implement these three methods.

## Reading the code

Read in this order. Each step needs only the steps before it.

1. `gm/ids.py`: canonical JSON and typed ids.
2. `gm/ledger/events.py`, `sqlite_event_log.schema.sql`, `sqlite_event_log.py`: events, ids, storage.
3. `gm/ledger/projection.py` and its indexes, then `gm/verify.py`.
4. `gm/policy.py`, then `gm/sampling.py`: what a decision sees, how it is made, and how the kernel picks samples.
5. `gm/controller.py`, `gm/settlement.py`, `gm/transitions.py`. Follow one action from `action-reserved` to its
   settlement.
6. `gm/run.py`, `gm/__main__.py`, then `domains/toy/` with `gm/domain/manifest.py`.
7. `gm/git_store/node_store.py`.
8. `gm/work_queue.py`, `gm/execution/runner.py`, `episodes.py`, `expansion.py`, `docker.py`, `gateway.py`.
9. `node/task_agents/shared/runtime.py` and `node/shared/gateway/client.py`: the container side.
10. `gm/execution/train.py`, if you use training from inside expansions.
