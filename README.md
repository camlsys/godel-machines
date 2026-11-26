# Gödel Machines

A general codebase for recursive self-improvement: systems whose agents rewrite their own code and keep a rewrite
only when it measurably improves. It implements the **Huxley-Gödel Machine (HGM)** and the **Red Queen Gödel
Machine (RQGM)**. Both search with the HGM policy (`gm.policy.Hgm`). HGM searches under a fixed evaluation
criterion. RQGM lets the evaluators co-evolve with the agents: at checkpoints, the form's declared `transitions`
promote judges (`objective_transition`), move curricula and score interpretations (`curriculum`), and mint
adversarial pools (`pool`) ([below](#hgm-or-rqgm)).

A **node** is a git tree holding a meta agent (the program that edits trees) and task agents (the programs that do
a domain's work). At each decision a search policy either asks a node's meta agent to write a child node (an
*expansion*), or evaluates a node: the kernel runs the node's task agents on a dataset sample and the domain's
judges score the result.

## Why build on it

- **Git and SQL as the substrate.** Every node is a deterministic commit in a git node store. Every decision and
  result is a content-addressed event in an append-only SQLite ledger. These two files are the whole run.
- **Every stored run replays exactly.** `gm verify` re-derives every event id and the checkpoint chain, then folds
  the ledger and prints a digest of the folded state. Every decision is seeded and keyed by its position in the
  ledger, so the ledger reproduces it.
- **Clear isolation.** Every episode is a fresh Docker container with all capabilities dropped and no network,
  or, when it may call a model, an internal network whose only route out is the host's metering gateway. Provider
  keys never enter a container. Task agents never see labels. Validation results are never shown to any agent;
  train results go only to the meta agent that submitted them.
- **Marking separated from agents.** The domain declares the judges. Each judgment runs in its own episode,
  apart from the work it scores, and is finished on the host by the form's `settle.py`, which lives outside the
  node and so outside any agent's reach. Every verdict is pinned to the judge, interpretation and dataset build it
  was made under.
- **One interface for training and validation.** From inside an expansion, the meta agent submits its current
  task agents on train samples with `rqgm-call submit`. The submission runs through the same reservations,
  runner, containers, judges and `settle.py` as the search's own validation evaluations, under its own ledger
  scope ([docs/agents.md](docs/agents.md#training-one-interface-for-training-and-validation)).
- **Extension points.** A domain is a folder: data, a `form.json`, a `settle.py` and a genesis tree
  ([docs/domains.md](docs/domains.md)). Agents are files in the node ([docs/agents.md](docs/agents.md)). A search
  policy is one object with `decide(snapshot) -> action` ([below](#your-own-search-policy)). An executor is any
  object implementing the three-method `WorkQueue` protocol in `gm/work_queue.py`
  ([docs/architecture.md](docs/architecture.md)).

## Quickstart

You need Linux with Docker and Python 3.14. On a Mac, run everything inside a Linux VM (for example colima). A
run records the source commit, so run it from a clean checkout and keep run stores outside it.

```sh
pip install -r requirements.txt
export PYTHONPATH=$PWD
images/build_meta_arm_images.sh python mini-swe          # the episode images the forms name
docker pull python:3.12-slim                             # image for the gateway relay container (--edge)
EDGE=$(docker image inspect python:3.12-slim --format '{{.Id}}')
```

Run the toy domain. It needs no model: a seeded scripted provider stands in for one.

```sh
python -m gm run --domain toy/mini_swe --store ../runs/toy --fresh --budget 6 --seed 0 --edge "$EDGE"
```

The run prints its id, event count, node count and `pick`, the node it returns (below). `--workers` (default 1)
is how many jobs run at once and `--cpus` (default `--workers`) the host CPUs they are admitted against. With one
worker the controller settles each job before its next decision, so two runs with the same seed write the same
ledger and the same node commits. With more, the ledger records completions in the order they arrive. There is
no configuration file; everything else a run uses is the form's `form.json` or a constant in code
([docs/domains.md](docs/domains.md#run-options)).

Run with a live model. The endpoint can be any OpenAI-compatible one; calls go through LiteLLM.

```sh
pip install litellm==1.97.0
python tools/operations/build_paper_corpus.py
export KEY=...
python -m gm run --domain paper/mini_swe --store ../runs/paper --fresh --budget 64 --seed 0 \
  --provider live --provider-url https://your-endpoint/v1 --provider-key-env KEY --edge "$EDGE"
```

The `bindings` in the form's `form.json` name the models: each `endpoint` is the LiteLLM model name sent to the
URLs given with `--provider-url`. Edit them for your models.

Verify a store:

```sh
python -m gm verify ../runs/toy            # or a bare ledger.sqlite; --json for machine output
```

## HGM or RQGM

Which machine a run is depends on one field of the form's `form.json`. With no transitions the evaluation criterion
never moves, and the run is HGM:

```json
"transitions": {}
```

With transitions declared, the judges, curricula, interpretations and adversarial pools move at checkpoints, and
the run is RQGM. Every shipped form declares transitions, so each runs RQGM as shipped:

| form | transitions |
| --- | --- |
| `toy/mini_swe` | `judge` (`objective_transition`), `dataset` and `interpretation` (`curriculum`), `adversarial` (`pool`) |
| `imo/mini_swe` | `grader` (`objective_transition`), `adversarial` (`pool`) |
| `paper/mini_swe` | `reviewer` (`objective_transition`), `adversarial` (`pool`) |
| `deepswe/mini_swe` | `reviewer` (`objective_transition`), `adversarial` (`pool`) |

To run any of them as HGM, set its `transitions` to `{}` and commit the change (a run records the source commit).
Checkpoints are still published, but no `criterion-transition` event is ever written. The rules themselves are in
`gm/transitions.py` and their fields in [docs/domains.md](docs/domains.md#formsformformjson).

## Your own search policy

A policy reads one frozen `Snapshot` and returns an `Expansion(parent)` or an `Evaluation(node, role, task)`; it
raises `NoAction` when it has nothing to issue now. The snapshot holds the archive (`parents`), each evaluation
cell's evidence as Beta pseudo-counts over fractional outcomes in [0, 1], the cells' load, the evaluations spent
and the `budget`, and seeded `draws` keyed by the run seed and the decision index. The policy never picks a
sample: the kernel chooses which sample an evaluation reads, so every record is valid whichever policy issued
it. `gm.run.run` runs one whole search as a function call and returns its report, including `pick`, the node with
the highest epsilon-best-belief `BB_eps = I^-1_eps(1 + S, 1 + F)` over its own evidence (`gm.policy.pick`).

```python
import os
from pathlib import Path

from gm.execution.providers import ScriptedProvider
from gm.execution.runner import Envelope
from gm.policy import Evaluation, Expansion, NoAction, evidence_maps
from gm.run import load_form, run


class Greedy:
    """Read the most promising node; expand it every third decision."""

    def decide(self, snapshot):
        own = evidence_maps(snapshot).node
        mean = lambda n: (own[n].score_sum + 1) / (own[n].trials + 2) if n in own else 0.5
        ranked = sorted(snapshot.parents, key=mean, reverse=True)
        if snapshot.action_index % 3 == 2 and snapshot.expandable:
            return Expansion(ranked[0])
        node = next((n for n in ranked if snapshot.open_cells(n)), None)
        if node is None:
            raise NoAction("nothing left to read")
        cell = snapshot.open_cells(node)[0]
        return Evaluation(node, cell.role, cell.task)


report = run(load_form("toy/mini_swe"), store=Path("../runs/greedy"), provider=ScriptedProvider(seed=0),
             envelope=Envelope(allotment="unbounded", nudge_at=None), budget=6, edge=os.environ["EDGE"], policy=Greedy())
print(report.nodes, len(report.event_ids), report.pick)
```

Without `policy=`, a run uses `gm.policy.Hgm`, the shipped HGM policy: Thompson draws over clade-pooled evidence
choose a parent to expand and over a node's own evidence a node to evaluate, and the tree widens while
`evaluations^alpha` reaches the published width ([docs/architecture.md](docs/architecture.md#the-policy-seam)).

## Data

- `toy`: committed in `domains/toy/data/`.
- `imo`: `python tools/operations/build_imo_corpus.py` downloads IMO-Bench (`proofbench.csv`, `gradingbench.csv`)
  at a pinned commit, checks the sha256 of each file and writes `domains/imo/data/build/`.
- `paper`: `python tools/operations/build_paper_corpus.py` downloads the hyperagents paper-review dataset at a
  pinned commit and writes `domains/paper/data/build/` for the papers listed in `split.json`.
- `deepswe`: `python tools/operations/build_deepswe_corpus.py` downloads DeepSWE at a pinned revision and the
  review patches `review.json` names, checks every sha256 and writes `domains/deepswe/data/build/`. Its solver
  works inside each task's own image, so it also needs `images/build_meta_arm_images.sh runtime`.

If a domain's data is missing, the run stops before it starts and prints the command to build it.

## Layout

| path | contents |
| --- | --- |
| `gm/` | CLI, `run.py` (start a run), controller, policy seam, sampling, settlement, transitions, verification |
| `gm/ledger/` | event codec, SQLite log, the fold and its indexes, the expansion-question digest |
| `gm/domain/` | reading a form and its data into one manifest |
| `gm/git_store/` | the git node store |
| `gm/execution/` | job runner, episodes, expansion publication, Docker containment, model gateway, providers, train pool |
| `node/` | the runtime copied into every root tree: gateway client, `rqgm-call`, task-agent runtime, `mini_swe` meta agent |
| `domains/` | `toy`, `imo`, `paper`, `deepswe` (data declarations, forms, genesis trees, settlement) and `common/` |
| `images/` | the build script for the episode images |
| `tools/operations/` | dataset builders for `imo`, `paper` and `deepswe` |

## Documentation

- [docs/architecture.md](docs/architecture.md): the run loop, the policy seam, what each module does, and an order
  for reading the code.
- [docs/domains.md](docs/domains.md): declaring a domain and a form; run options.
- [docs/agents.md](docs/agents.md): writing your own meta agent and task agents, training from inside an
  expansion, how episodes reach a model, and episode containment.
- [docs/challenge.md](docs/challenge.md): comparing agents on one domain.
- [docs/ledger.md](docs/ledger.md): ledger tables, events, ids, artifacts, the checkpoint chain, the fold.
- [docs/nodes.md](docs/nodes.md): the git node store format.

## License

Apache-2.0.
