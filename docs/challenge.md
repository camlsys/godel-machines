# Running a challenge

In a challenge, several participants start from the same domain. Each brings their own meta agent and task
agents, runs the same search budget, and the results are compared. A run's store holds everything needed to check
it, so participants exchange stores, not reports.

## Fix the task

The domain's `data/`, its `settle.py`, and the task half of each `form.json` define the task and how it is scored:
`domain`, `roles` (tasks, sets and `reads`), `judges`, `coordinates`, `pins`, `interpretations` and `transitions`.
All participants must use identical copies of these. Pick one of the shipped domains or declare a new one
([domains.md](domains.md)). Build its data first (see the README). `toy` needs no model, so it is useful for a dry
run.

## Bring your agent

Each participant edits only the method half:

- `domains/<domain>/forms/<form>/genesis/`: your task agents under `task_agents/<role>/`, and
  `meta/instruction.txt`, which steers the shipped meta agent ([agents.md](agents.md)).
- In `form.json`: `bindings` (your models), `weights`, the `expansion` episode and each role's `episode` (`argv`,
  `image_recipe`, `tag`, `token_allotment`), and the form's `cpus` and `memory_bytes`.

Agree on the bindings, token allotments, `--budget`, `--workers` and seeds before anyone runs. These affect cost and
results as much as the agent code does. `--workers 1` makes each run reproducible from its seed.

Build the episode images (`images/build_meta_arm_images.sh`) and commit your fork. A run records the HEAD of the
source checkout in its first event (`source_commit`) and refuses to start from a checkout with uncommitted changes.
Keep run stores outside the checkout, or the first run will leave the tree dirty for the next one.

## Run

```sh
export PYTHONPATH=$PWD
EDGE=$(docker image inspect python:3.12-slim --format '{{.Id}}')
for s in 0 1 2; do
  python -m gm run --domain <domain>/<form> --store ../runs/<name>-s$s --fresh --budget <N> --seed $s \
    --workers 1 --provider live --provider-url https://<endpoint>/v1 --provider-key-env KEY --edge "$EDGE"
done
```

`--budget` is the number of search evaluations a run may count. A single run is noisy, so use several seeds.

## Check and compare

```sh
python -m gm verify ../runs/<name>-s0 --json
```

`verify` re-encodes every event and checks its id and the checkpoint chain. A changed payload no longer matches its
id, and a changed id no longer matches the prefix digest of any later checkpoint. To compare runs, send
`ledger.sqlite` and `nodes/` for each run.

The ledger is plain SQLite. Each `judgment-recorded` event scores one `task-outcome`, and the outcome names the
node and role. Training from inside expansions is recorded too, under scope `train`, so join the judgment's
`action-reserved` event to keep search evaluations only. This query gives the mean utility per node, role and
judge:

```sql
select o.payload->>'$.node' node, o.payload->>'$.role' role, j.payload->>'$.slot' judge,
       count(*) n, avg(j.payload->>'$.judgment.outcome') mean
from events j
join events o on o.event_id = j.payload->>'$.subject.outcome'
join events r on r.variant = 'action-reserved' and r.payload->>'$.action' = j.payload->>'$.action'
where j.variant = 'judgment-recorded' and r.payload->>'$.scope' = 'search'
group by 1, 2, 3;
```

Both event kinds carry the `pins` they were scored under: dataset build, interpretation, judge and sample.
Restrict to the final pins before you compare runs whose judges or datasets moved during the search. The sets a
role is evaluated on are its tasks' `search` sets in `form.json`; training uses their `train` sets.

Every node the search produced is in the node store. To see how a node was reached:

```sh
git --git-dir ../runs/<name>-s0/nodes log refs/rqgm/nodes/<node>
```
