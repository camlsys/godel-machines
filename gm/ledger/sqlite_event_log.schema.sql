-- Purpose: the durable record of one run. Every fact the controller may later
-- rely on arrives here as an append-only event; every other table is either a
-- content store those events point into, or a small piece of working state the
-- fold rebuilds and compares against. Nothing here is derived state a reader
-- may trust without refolding: the events are the record, the rest is
-- bookkeeping.
--
-- Materialized form. Arrows point from the referencing column to its target.
--
--   schema_version(version)                    -- one row, the migration level
--
--   events                                     artifacts
--   ------------------------------             ------------------------
--   seq            INTEGER PK AUTOINC          oid    TEXT PK
--   event_id       TEXT UNIQUE                 data   BLOB
--   payload        TEXT                                  ^
--   variant        TEXT GENERATED STORED                 |
--       ^   ^                                            |
--       |   |   event_artifacts                          |
--       |   |   ------------------------------           |
--       +---|-- event_seq  INTEGER  PK part -------------+  (via oid)
--           |   path       TEXT     PK part
--           |   inline     BLOB
--           |   oid        TEXT ----------------------->  artifacts.oid
--           |   size       INTEGER
--           |
--   datasets                                   dataset_sets
--   ------------------------------             ------------------------------
--   version        TEXT PK  <----------------- version  TEXT PK part
--                                              name     TEXT PK part
--       ^                                      ordinal  INTEGER PK part
--       |                                      sample   TEXT
--       |
--       |          dataset_files                        dataset_images
--       |          ------------------------------       ---------------------
--       +--------- version        TEXT PK part          version TEXT PK part
--       |          sample         TEXT PK part          sample  TEXT PK part
--       |          channel        TEXT PK part          image   TEXT
--       |          ordinal        INTEGER PK part            |
--       |          oid            TEXT ---> artifacts.oid    |
--       +--------------------------------------------------+  (via version)
--
--   reservations                               completion_inbox
--   ------------------------------             ------------------------------
--   action        TEXT PK  <------------------ action   TEXT UNIQUE
--   reserved_seq  INTEGER --> events.seq       arrival  INTEGER PK AUTOINC
--   settled_seq   INTEGER --> events.seq       payload  TEXT
--       ^
--       |   retry_state                        checkpoint_phase
--       |   ------------------------------     ------------------------
--       +-- action      TEXT PK               id     INTEGER PK, = 1
--           attempts    INTEGER               phase  TEXT
--           last_error  TEXT                  scope  TEXT
--           spend_rows  TEXT                  fired  TEXT
--
-- Exclusive and paired constraints, stated in prose because a diagram cannot
-- carry them:
--   * event_artifacts stores a payload byte-string in exactly one place. Either
--     inline holds it or oid points at an artifact, never both and never
--     neither.
--   * checkpoint_phase holds exactly one row, pinned by id = 1.
BEGIN IMMEDIATE;
CREATE TABLE schema_version (
-- Migration level this store was written at; one row, compared on open.
version INTEGER NOT NULL
) STRICT;
CREATE TABLE events (
-- Arrival order of the append-only record. Gaps would mean a lost event.
seq INTEGER PRIMARY KEY AUTOINCREMENT,
-- Content hash over the canonical payload bytes; identity, not a sequence.
event_id TEXT NOT NULL UNIQUE,
-- Canonical JSON of the event. The record itself, never rewritten.
payload TEXT NOT NULL,
-- Extracted from the payload so a reader can filter without parsing every row.
variant TEXT GENERATED ALWAYS AS (json_extract(payload, '$.variant')) STORED
) STRICT;
CREATE TABLE artifacts (
-- Content address of the bytes; equal bytes are stored once.
oid TEXT PRIMARY KEY,
-- The bytes themselves, or empty when the payload is materialised beside the
-- ledger instead. Over `INLINE_CAP` a row stops being a good way to hold bytes
-- -- SQLite reads a blob whole, and refuses one past a gigabyte by raising in
-- the middle of an append -- so the row stays as the index and the payload
-- takes the path a filesystem is good at. Every artifact still has exactly one
-- row, so a reader asking whether an address exists asks one question, and the
-- shape is unchanged so no recorded run has to be reopened at a new version.
data BLOB NOT NULL
) STRICT;
CREATE TABLE event_artifacts (
-- The event these bytes belong to.
event_seq INTEGER NOT NULL REFERENCES events(seq),
-- Path the bytes occupied, unique within one event.
path TEXT NOT NULL,
-- Every payload lives in artifacts and is named here. There is no inline
-- column: bytes never travel inside the event write path, which is what
-- keeps an artifact-bearing event as cheap to append as a reference.
oid TEXT NOT NULL REFERENCES artifacts(oid),
-- Byte length, so a consumer can size a payload without reading it.
size INTEGER NOT NULL,
PRIMARY KEY (event_seq, path)
) STRICT;
CREATE TABLE datasets (
-- Declared dataset version; the pin every sample row hangs from.
version TEXT PRIMARY KEY
) STRICT;
CREATE TABLE dataset_sets (
-- Dataset version this membership belongs to.
version TEXT NOT NULL REFERENCES datasets(version),
-- Declared set name, such as a train or holdout split.
name TEXT NOT NULL,
-- Position within the set; membership is ordered, not a bag.
ordinal INTEGER NOT NULL,
-- Sample key admitted to the set at this position.
sample TEXT NOT NULL,
PRIMARY KEY (version, name, ordinal)
) STRICT;
CREATE TABLE dataset_files (
-- Dataset version this file belongs to.
version TEXT NOT NULL REFERENCES datasets(version),
-- Sample the file belongs to.
sample TEXT NOT NULL,
-- Declared channel, such as an input the candidate may read.
channel TEXT NOT NULL,
-- Position within the channel; file order is declared, not incidental.
ordinal INTEGER NOT NULL,
-- Content address of the file bytes.
oid TEXT NOT NULL REFERENCES artifacts(oid),
PRIMARY KEY (version, sample, channel, ordinal)
) STRICT;
CREATE TABLE dataset_images (
-- Dataset version this image pin belongs to.
version TEXT NOT NULL REFERENCES datasets(version),
-- Sample the image runs for.
sample TEXT NOT NULL,
-- Pinned runtime image reference for this sample.
image TEXT NOT NULL,
PRIMARY KEY (version, sample)
) STRICT;
CREATE TABLE reservations (
-- Action identity; one row per issued action, settled or not.
action TEXT PRIMARY KEY,
-- Event that reserved the action.
reserved_seq INTEGER NOT NULL REFERENCES events(seq),
-- Event that settled it. NULL means still in flight.
settled_seq INTEGER REFERENCES events(seq)
) STRICT;
CREATE TABLE completion_inbox (
-- Arrival order of completions awaiting application.
arrival INTEGER PRIMARY KEY AUTOINCREMENT,
-- Reserved action this completion settles; one pending completion per action.
action TEXT NOT NULL UNIQUE REFERENCES reservations(action),
-- Canonical JSON of the completion, applied in arrival order.
payload TEXT NOT NULL
) STRICT;
CREATE TABLE checkpoint_phase (
-- Pinned to one, so this table holds a single row of working state.
id INTEGER PRIMARY KEY CHECK (id = 1),
-- Where the checkpoint machine currently stands. Written by a guarded update
-- naming the phases it may displace, so this cell is a compare-and-swap and
-- not a blind assignment; `CheckpointPhase.displaces` owns the edge set.
phase TEXT NOT NULL,
-- Scope the current phase applies to, when one is in force.
scope TEXT,
-- What fired the current phase, when it was triggered rather than entered.
fired TEXT
) STRICT;
CREATE TABLE retry_state (
-- Reserved action this retry bookkeeping belongs to.
action TEXT PRIMARY KEY REFERENCES reservations(action),
-- Attempts made so far against the declared ceiling.
attempts INTEGER NOT NULL,
-- Last failure text, kept so a retry decision can cite it.
last_error TEXT NOT NULL,
-- Spend already charged by earlier attempts, so a retry cannot double-count.
spend_rows TEXT NOT NULL
) STRICT;
-- Must equal `dataset_store._SCHEMA_VERSION`, which `_require_schema_version` compares on open.
-- The two are written apart and read together, so a revert touching only the constant leaves
-- every freshly created store refusing itself at the version it was just written at.
INSERT INTO schema_version (version) VALUES (12);
INSERT INTO checkpoint_phase (id, phase) VALUES (1, 'idle');
