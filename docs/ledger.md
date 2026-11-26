# Ledger format

A run directory holds two durable things: `ledger.sqlite` (this page) and `nodes/` (the git node store,
[nodes.md](nodes.md)). Everything else the code computes is a fold of the ledger.

## Tables (SQLite, store schema 12)

The DDL is `gm/ledger/sqlite_event_log.schema.sql`. All tables are `STRICT`.

| table | contents |
| --- | --- |
| `schema_version(version)` | one row, `12` |
| `events(seq, event_id, payload, variant)` | the append-only record; `variant` is generated from `payload` |
| `artifacts(oid, data)` | content-addressed bytes, `oid = sha256(data)`; bytes over 2 MiB live beside the ledger in `ledger.sqlite-artifacts/`, zstd-framed when that is smaller, and keep an empty row |
| `event_artifacts(event_seq, path, oid, size)` | which artifact bytes an event carries, under which path |
| `datasets`, `dataset_sets`, `dataset_files`, `dataset_images` | the dataset builds a run samples from, registered before its first event |
| `reservations(action, reserved_seq, settled_seq)` | one row per reserved action; `settled_seq` null while in flight |
| `completion_inbox`, `checkpoint_phase`, `retry_state` | small working tables written in the same transaction as events |

Only `events`, `artifacts` and `event_artifacts` are the record. The rest is bookkeeping a run writes alongside. A
store written at schema 11, whose `datasets` rows also carry a provenance column, opens as it is.

## Events

`payload` is canonical JSON: keys sorted, no whitespace (`separators=(",", ":")`), no NaN or infinity
(`gm/ids.py: canonical_json`). Every payload carries `"rqgm-event"`, the event schema it is spelled in (10 for a run
this code starts), and a `variant`. The event id is `sha256(payload)` in hex, so equal facts have equal ids.

The event union (`gm/ledger/events.py`):

| variant | written when | main fields |
| --- | --- | --- |
| `run-configured` | first event of every run | `manifest`: `run`, `form`, `seed`, `workers`, `budget`, `source` (the root node), `source_commit`, `pins`, `coordinates`, `snapshot_sets`, `seat_interpretations`, `bindings`, `tags`, `weights`, `images` (each episode's image digest), `train` |
| `action-reserved` | the controller decided to do something | `action` (`a1`, `a2`, ...), `kind` (`expansion`, `generation`, `judgment`), `node`, `role`, `task`, `scope` (`search`/`train`), `sample`, `owed`, `slot` |
| `expansion-recorded` | an expansion episode settled | `action`, `parent`, `child` (new node or null), `result`, `experience_v4` (the question digest), `attempt_no`, `arm` (the episode binding), `outcome` (`published`, `duplicate`, `declined`, `failed`) |
| `task-outcome` | a generation episode settled | `action`, `node`, `role`, `task`, `pins`, `result` |
| `judgment-recorded` | a judgment settled | `action`, `slot`, `subject` (an outcome, or a labeled node and sample), `pins`, `judgment` |
| `soft-skip` | an action exhausted its infrastructure retries | `action`, `attempts`, `error`, `spend`, `dialogues` |
| `node-state-recorded` | an episode's first turn landed (`admitted`), or an expansion ended (`final_turn`) | `action`, `node`, `state`, `reason`, `done`, `grounds` |
| `criterion-transition` | a checkpoint moved a criterion coordinate | `coordinate`, `incumbent`, `replacement`, `epoch`, `minted` |
| `checkpoint-published` | a checkpoint was crossed | `last_event`, `prefix_digest`, `trigger`, `scope` |

The run's identity is the id of its `run-configured` event: every setting a decision reads is either in that
manifest or a constant in code. Some optional fields are left out of the payload when unset instead of being
written as null: `slot` on `action-reserved`, `outcome` and `arm.idle_seconds` on `expansion-recorded`, and
`reason`, `done` and `grounds` on `node-state-recorded`.

Every result carries its `spend`: rows of (binding tag or null, axis, amount). Settlement adds one `evaluations`
unit to each counted trial: a failed generation, or a judgment that closes its cover. That axis is the budget.

## Schema-8 ledgers

Ledgers written by the previous event schema (`"rqgm-event": 8`) read through the same codec. Their
`run-configured` manifest carries fields this schema does not name (the resolved configuration, resources,
ceilings, identity pins and others), and their `expansion-recorded.arm` carries a few more. The codec keeps those
fields verbatim beside the typed ones and writes them back, with version 8, on re-encode, so every stored event
reproduces its bytes and its id. Decisions on such a ledger read the settings it recorded
(`gm/ledger/settings.py`).

## Artifacts

An event that carries files (episode outputs, dialogues, transcripts, file effects) lists them in its payload as
`"artifacts": [[channel, path, oid, size], ...]`, sorted, with `channel` either `files` (outputs the domain
declared) or `metadata` (host evidence). The bytes live in `artifacts`; `event_artifacts` repeats the
(event, path, oid, size) rows for lookup. Because the list is inside the payload, the artifact bytes are part of
the event id.

## Checkpoint chain

`checkpoint-published` pins the prefix it closes:

```
h0     = sha256(b"rqgm-prefix")
h(i+1) = sha256(h(i) || event_id(i).encode())      # raw digest bytes, then the hex id as text
```

`prefix_digest` is `h` over every event before the checkpoint, and `last_event` is the id of the event just before
it. `gm verify` recomputes both. A checkpoint is due after 1, 2, ..., 9, 11, 13, ... counted evaluations
(`amortized_threshold` in `gm/transitions.py`), and again at the end of a run whose own transitions invalidated an
earlier comparison.

## The fold

`gm/ledger/projection.py` folds events one at a time into a `Fold`: lineage and publication order, pins and
epochs, open actions with the pins each was issued under, outcomes, judgments, spend, minted sets, and the indexes
decisions read. The controller keeps one fold and applies each new event to it, so a run costs one fold step per
event; `project(events)` folds a whole ledger once for offline reads. At each checkpoint that moves a criterion,
the controller also refolds the whole ledger and checks that the result equals its incremental fold.

## Verifying a ledger

```sh
python -m gm verify <store-or-ledger.sqlite> --json
```

checks, in one pass: each payload decodes and re-encodes to the same text, that text hashes to the stored id, each
checkpoint matches the chain, and the whole ledger folds. It prints a `projection` digest over the folded facts
(lineage, pins, epochs, decisions, open actions, done nodes, outcomes, judgments, spend, completed cells, minted
sets), so two implementations can compare folds ledger by ledger. It exits 1 on any problem. Cost is linear in the
number of events.
