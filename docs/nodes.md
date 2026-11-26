# Node store format

`<store>/nodes` is a bare git repository written with pygit2 (`gm/git_store/node_store.py`). A node is a commit.
Its tree is the whole agent program: meta agent, task agents, shared runtime, domain files and the experience the
host wrote for it.

## Determinism

- Every commit has the fixed author and committer `RQGM <rqgm@example.invalid>`, with timestamp 946684800 and
  offset 0. A node id therefore depends only on its tree, parents and message.
- The repository config is pinned: `core.autocrlf=false`, no excludes file, file modes and symlinks on, no
  signing, hooks and gc off.
- The root node is the genesis tree ([domains.md](domains.md#the-genesis-tree)), committed with no parent. Its id is
  `source` in the `run-configured` manifest. `create_root` can be called again for another root; nothing assumes
  there is only one.

## Refs

| ref | contents |
| --- | --- |
| `refs/rqgm/nodes/<commit>` | one ref per published node |
| `refs/rqgm/turns/<key>` | the open turn chain of an expansion in progress; `key` is the sha256 of `turn:<action>` |

There are no other index files. A node's lineage is its first-parent history restricted to commits that have a node
ref (`lineage`), and its depth is the length of that lineage minus one. The ledger records which action produced
which node.

## Turn chains

An expansion opens a chain on its parent (`open_chain`) and moves the ref `refs/rqgm/turns/<key>` with every commit
on it:

- After each model call the gateway serves for the expansion, the host copies the container's `/workspace` out and
  commits it with the message `model call <n>`. The commit carries the dialogue so far in
  `experience/generations/<depth>/node-dialogue.md`. A call that changed no file adds no commit.
- Each train submission from inside the episode commits the current tree with the message
  `train submit <n>`, followed by `: <tag>` when the submission named a tag.
- When the episode ends, `seal` commits the final tree on top of the chain, deletes the turn ref and creates the
  node ref.

Any commit messages the agent wrote since the last commit come first in the next one. The chain is the child's
first-parent history, so an observer can walk every turn of the expansion that produced a node:

```sh
git --git-dir <store>/nodes log --first-parent --format='%h %s' <parent>..refs/rqgm/nodes/<child>
```

`seal` refuses a tree that differs from the parent only in host-written experience files (the expansion is
declined), and refuses a node ref that already exists (a duplicate of a published node).

## Files

Node files are ordinary git blobs. When a tree is committed, every entry must be a regular file or symlink, no file
may exceed 512 MiB (`MAX_FILE_BYTES`) and the tree may not exceed 4 GiB in total (`MAX_NODE_BYTES`).

## Tree layout

| path | written by | contents |
| --- | --- | --- |
| `meta/` | genesis, then the meta agent | the meta agent: `node/mini_swe/` plus the form's `meta/instruction.txt` |
| `task_agents/<role>/` | genesis, then the meta agent | one package per role ([agents.md](agents.md#task-agents)) |
| `task_agents/shared/` | genesis | the task-agent runtime, its I/O contract check and `TRAIN.md` |
| `shared/` | genesis | the gateway client and `rqgm-call` |
| `experience/data/train/<sample>/` | host | each train sample's input files and its `labels/` |
| `experience/generations/<depth>/` | host and meta agent | one folder per expansion on the chain: `results.csv`, `summary.json`, `evaluations/<action>/`, `node-dialogue.md`, `prompt.json` |
| `.rqgm/` | host | per-episode control files: `contract.json`, `job.json`, `environment.json` |

The paths the host writes are spelled once, in `gm/experience.py`. The root carries an empty
`experience/generations/0/` with a `README.md`; every descendant keeps its ancestors' folders beside its own.

## What an episode receives

- An expansion gets the parent with its full git history (`experience_view`). The host links the train data into
  `experience/data/` and creates `experience/generations/<depth>/` for the new node. These files are committed with
  the child, so descendants can read them.
- A generation or judgment gets `materialize_task(node)`: the node's history with `experience/` and host control
  files removed from every commit. Task agents never see labels or training results.

Outputs, evidence and scores are stored in the ledger, not in git.
