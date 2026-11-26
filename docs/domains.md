# Domains and forms

A domain is a folder, `domains/<name>/`. Nothing is registered: `gm/run.py` (`forms()`) lists every
`domains/<domain>/forms/<form>/` that holds a `form.json` and a `genesis/` folder, and `--domain <domain>/<form>`
names one of them. The smallest complete domain is `domains/toy/`. Copy it to start a new one.

```
domains/<domain>/
├── data/
│   ├── fetch.json       how the CSV sets are read, and the builder that writes them
│   └── <set>.csv        one file per sample set, committed or built
└── forms/<form>/
    ├── form.json        the whole declaration: roles, judges, transitions, episodes, models, limits
    ├── settle.py        turns a judgment's files into a score
    └── genesis/         the domain layer of the root tree
```

One reader turns all of this into one frozen `DomainManifest`: `read_form` in `gm/domain/manifest.py` reads
`form.json` and composes the root tree, and `read_dataset` in `gm/domain/dataset.py` reads `data/`.
`load_form` in `gm/run.py` first replaces every `image_recipe` with the image digest the local Docker daemon holds
for it (`resolved_images` in `gm/execution/docker.py`).

## data/fetch.json

| key | meaning |
| --- | --- |
| `key` | the CSV column that names a sample |
| `channels` | CSV column to channel name. A channel named `input` is staged as the sample's input files; the others are kept per sample by name |
| `sets` | set name to CSV path, relative to `data/` |
| `version` | the name of this dataset build; it is the run's initial `dataset` pin |
| `successors` | later builds (version to its own `sets`) that a curriculum transition can switch to |
| `build.tool` | the builder script. A run whose CSV files are missing stops and names it |
| `image` | optional: the channel that names each sample's own Docker image |
| `private` | optional: columns left out of the train rows staged into nodes |

`hidden`, `identity` and `quarantine` also appear in the shipped files. They record how the builders split and
label the data; `gm/` does not read them.

What a node can see of a sample is decided elsewhere. A search or train episode receives only the channel its role
`reads`. Train samples are also staged into expansions under `experience/data/train/<sample>/`: the input files,
plus every CSV column except the `private` ones under `labels/` ([nodes.md](nodes.md)). Settlement receives the
sample's whole channel row.

## forms/<form>/form.json

| key | meaning |
| --- | --- |
| `domain` | the run id |
| `roles.<role>.tasks` | the role's tasks; each task names its `search` set (validation during the search) and its `train` set (training submitted from an expansion, see [agents.md](agents.md)) |
| `roles.<role>.reads` | the channel the role is staged: a channel name, `input` for the sample's input files, or `null` for none |
| `roles.<role>.episode` | the role's episode (below) |
| `judges.<slot>` | one judge slot. A slot named like a role is a node judge: that role's task agent in the node scores the work, then `settle.py <slot>` finishes the score. Any other slot is fixed: only `settle.py <slot>` runs |
| `judges.<slot>.over` | the roles this slot judges |
| `judges.<slot>.rule` | the scoring rule id recorded with each verdict |
| `judges.<slot>.reads` | the produced files the judge is staged |
| `judges.<slot>.prepare` | optional: channel to file name, extra channels staged to a node judge as files |
| `judges.<slot>.settles` | the hidden channels the slot is settled against (declared; `settle.py` receives the whole channel row) |
| `judges.<slot>.identity` | whether a fixed judge is told the sample key |
| `judges.<slot>.deadline_seconds` | the time limit for `settle.py` |
| `judges.<slot>.interpretation`, `.tag` | optional: an interpretation fixed for this slot, and the model binding its spend is attributed to |
| `coordinates` | the criterion coordinates: which names the `sample`, `dataset` and `interpretation` axes, and `currency` (whether evidence counts only under the coordinate's current value) |
| `pins` | the initial value of coordinates that are not set from the data or the root node |
| `interpretations` | name to kind: `threshold` (`bar`), `band` (`low`, `high`), `full_credit` or `graded`. `domains/common/settlement.py` (`utility`) turns a raw score into a utility under each |
| `transitions.<coordinate>` | how a coordinate moves at checkpoints: `curriculum` (`at` a clock value, `source` to `target`), `objective_transition` (promote the judge whose evidence on the declared `surfaces` is best, after `min_trials`), `pool` (mint a set named `name` from judgments that meet `criterion`, keyed by `signer`) |
| `expansion` | the meta agent's episode |
| `bindings.<tag>` | one model: `endpoint` (the LiteLLM model name), optional `effort` and `kwargs`, `output_tokens`, `sequence_tokens` |
| `weights.<tag>` | the weight of each token axis when a grant meters a binding's calls |
| `cpus`, `memory_bytes` | Docker limits for every episode of the form |

An episode declares `argv`, `image_recipe` (`meta-arm/python`, `meta-arm/mini-swe` or `meta-arm/runtime`, built by
`images/build_meta_arm_images.sh`), `tag` (a binding, or `null` for no model), `outputs` (declared name to the path
the episode writes), `evidence` (extra files recorded), `token_allotment` and `idle_seconds`. An optional `checkout`
runs the episode inside the sample's own image, with the agent runtime copied in; deepswe's solver uses it.
[agents.md](agents.md) explains how `tag` decides whether an episode can reach a model.

Checkpoints are not declared. They follow the amortized schedule in `gm/transitions.py`
(`amortized_threshold`): after 1, 2, ..., 9, 11, 13, ... counted evaluations, each threshold about 9/8 of the last.

## Run options

```sh
python -m gm run --domain <domain>/<form> --store <dir> --fresh --budget N --seed S [--workers W] [--cpus C] --edge <image>
```

| option | meaning |
| --- | --- |
| `--budget` | the number of search evaluations the run may count |
| `--seed` | the seed every policy draw is keyed by |
| `--workers` | concurrent jobs, default 1. With 1 the controller settles each job before the next decision, so a run is reproducible from its seed |
| `--cpus` | the host CPUs jobs are admitted against, default `--workers` |

`--provider`, `--provider-url`, `--provider-key-env`, `--surface`, `--episode-tokens` and `--nudge-at` choose the
model provider and what episodes are told about their token budget. `--edge` is the image of the gateway relay
container. There is no configuration file.

The rest is fixed in code:

| constant | where |
| --- | --- |
| clock axis `evaluations`, 1 unit held per open evaluation | `CLOCK`, `CLOCK_RESERVE` in `gm/ledger/events.py` |
| at most 1 open action per (kind, sample set) cell | `IN_FLIGHT` in `gm/sampling.py` |
| HGM `alpha` = 2/3, bootstrap cohort floor(workers^alpha) | `Hgm` in `gm/policy.py` |
| re-asks of a question a parent answered without a child: min(max(5, 3·workers), 64) | `settings` in `gm/ledger/settings.py` |
| the final pick: highest BB_eps over a node's own evidence, eps = 0.05 | `pick` and `EPSILON` in `gm/policy.py` |
| 2 attempts for a job that fails on infrastructure | `Runner` in `gm/execution/runner.py` |

## forms/<form>/settle.py

The host runs `python settle.py <slot>` in the form's folder (`ProcessInstrument` in `gm/domain/seams.py`). It
writes a JSON list of members on stdin: each with the staged `files` (base64) or, after a node judge, its raw
`score`, plus the sample's `channels`. The program answers one verdict per member on stdout. `domains/common/settlement.py`
holds the shared wire: a form supplies `raw_score` and `meant` to `Law`, which reads `form.json` beside it and
fills in the utilities for every declared interpretation. A non-zero exit makes the judgment invalid, recorded with
the error; `Law` exits 75 when it cannot read its input.

## The genesis tree

`compose` in `gm/domain/manifest.py` builds the root tree from three layers. When two layers write the same path,
the later one wins.

1. From `node/`: `shared/` (gateway client, `rqgm-call`), `task_agents/` (the task-agent runtime), `tests/` and
   `pytest.ini`.
2. `node/<form>/`, placed under `meta/`. Every shipped form is named `mini_swe`, so it gets `node/mini_swe/`. Its
   `harness.json` is replaced by the upstream files it pins (fetched once and checked by sha256) and an
   `upstream.json` record.
3. `domains/<domain>/forms/<form>/genesis/`: the domain's task agents, `meta/instruction.txt`, seat tests,
   requirements.

`gm/genesis_tree.py` then adds `experience/generations/0/README.md` and the train data under `experience/data/`.
Every byte of the result is part of the root node's id.
