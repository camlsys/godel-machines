# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Build the `imo` corpus under `domains/imo/data/build/` from IMO-Bench.

Downloads `proofbench.csv` and `gradingbench.csv` from google-deepmind/superhuman at a pinned commit,
checks their sha256, derives the filtered grading set, assigns every problem to one of train,
validation or heldout (the problems in `quarantine.json` are always heldout), and writes the sets
`fetch.json` names. Every written member is checked against its pinned sha256.

    python tools/operations/build_imo_corpus.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "domains" / "imo" / "data"
UPSTREAM = "https://raw.githubusercontent.com/google-deepmind/superhuman/c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8/imobench/"
SOURCES = {
    "proofbench.csv": "c03ad2d9788ba147022031d90ecf192d16c038939eb432759914902598a87d6d",
    "gradingbench.csv": "e85a520c2bbb5a89f2db35088c7935d485922f26cf2bbd04f15e1d967af26cfe",
}
FILTERED_SHA256 = "a1fa39630027ba6eb93761c8b97753e62d42a580b6da3656f828f6509e56f7b9"
MEMBERS = {
    "heldout": "02f9cb99ffb0cf76be4289bc1ab7be530c25480af80662cdb85cabdaca8d7972",
    "proof_heldout": "337384a8e6eeb3b053f4e7d818b5b90c47e7e8bec8ce61402c5ec08010c5724a",
    "proof_train": "ea6ef0ff4f235bc64b67de6ddfb7efa46fafeeddab743ac6a606771eb72dde46",
    "proof_validation": "6fa5c8ab6239f0a7be8c110f3af2f2fe1577c2b8c93ff74a9861e5afe7345acd",
    "train": "908541f0735737f0b0b3929bbc0747295a719a9a7c9a1f1c525a52f36d4cacb5",
    "train_negative": "1ba3cc25f55b9fda636cca474a8ac01f9508909dbfc4a2688a8489bcaf0b3f92",
    "train_positive": "4e48abcd2917c223292a738283d5d44a02eb50519d3261107266c0f491ad3e30",
    "validation": "9eac34b15e9c91a2f7df7c223e0e72b2f4bf48f7240957e709a1e25c136637d1",
    "validation_negative": "2113ac6a78c682acfbf54671bde68f490c1ee2bcf8951cdfe92a7f26d5ed4a9f",
    "validation_positive": "bb265399606923a0c2304c7b3881aa224466bc743db09e1121907bb5f1a20097",
}
MARKS = frozenset({"0", "1", "6", "7"})
LEVELS = ("pre-IMO", "IMO-easy", "IMO-medium", "IMO-hard")
SPLITS = ("train", "validation", "heldout")
SHARED = ("Problem", "Solution", "Grading guidelines")
TRUNCATED = 49999
BAR = 7.0
COLUMNS = ("Grading ID", "Problem ID", "Problem", "Response", "Marking", "Points")
PROOF_COLUMNS = ("Grading ID", "Problem ID", "Problem", "Solution", "Grading guidelines", "Guidance")

csv.field_size_limit(sys.maxsize)


def fetched(name: str) -> str:
    """Download one upstream member and check its sha256."""
    with urllib.request.urlopen(UPSTREAM + name, timeout=120) as response:
        body = response.read()
    if (seen := hashlib.sha256(body).hexdigest()) != SOURCES[name]:
        raise SystemExit(f"{UPSTREAM}{name} has sha256 {seen}, expected {SOURCES[name]}")
    return body.decode("utf-8")


def permutation(n: int, seed: int) -> list[int]:
    """Return `numpy.random.RandomState(seed).permutation(n)`, computed with the standard library."""
    state = [seed & 0xFFFFFFFF]
    for i in range(1, 624):
        state.append((1812433253 * (state[-1] ^ (state[-1] >> 30)) + i) & 0xFFFFFFFF)
    rng = random.Random()
    rng.setstate((3, (*state, 624), None))
    order = list(range(n))
    for i in reversed(range(1, n)):
        mask = i
        for shift in (1, 2, 4, 8, 16):
            mask |= mask >> shift
        while (j := rng.getrandbits(32) & mask) > i:
            pass
        order[i], order[j] = order[j], order[i]
    return order


def filtered(grading: str) -> str:
    """Keep responses marked 0, 1, 6 or 7 and shuffle them with seed 42, as upstream curates them."""
    reader = csv.reader(io.StringIO(grading))
    header = next(reader)
    points = header.index("Points")
    kept = [row for row in reader if row[points] in MARKS]
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(kept[i] for i in permutation(len(kept), 42))
    text = out.getvalue()
    if hashlib.sha256(text.encode()).hexdigest() != FILTERED_SHA256:
        raise SystemExit("the filtered grading set does not match its pinned sha256")
    return text


def fenced(row: dict[str, str], parts: tuple[str, ...]) -> str:
    """Return the named columns of one row as `### <name>` sections."""
    return "\n\n".join(f"### {name}\n\n{row[name]}" for name in parts)


def joined(proof: list[dict[str, str]], grading: list[dict[str, str]]) -> list[dict[str, str]]:
    """Join grading rows onto their problems, dropping responses cut at the upstream length limit."""
    problems = {row["Problem ID"]: row for row in proof}
    return [
        {
            "Grading ID": row["Grading ID"],
            "Problem ID": row["Problem ID"],
            "Problem": problems[row["Problem ID"]]["Problem"],
            "Response": row["Response"],
            "Marking": fenced({**problems[row["Problem ID"]], "Response": row["Response"]}, (*SHARED, "Response")),
            "Points": row["Points"],
        }
        for row in grading
        if len(row["Response"]) != TRUNCATED
    ]


def graded_split(marks: dict[str, list[str]], levels: dict[str, str]) -> dict[str, str]:
    """Split the graded problems in two halves that both carry every mark, balancing difficulty then marks."""
    problems = sorted(marks)
    choices = []
    for left in combinations(problems, len(problems) // 2):
        if len(problems) % 2 == 0 and problems[0] not in left:
            continue
        right = tuple(pid for pid in problems if pid not in left)
        histograms = [Counter(mark for pid in group for mark in marks[pid]) for group in (left, right)]
        if any(set(counts) != MARKS or max(counts.values()) > counts.total() / 2 for counts in histograms):
            continue
        difficulty = [Counter(levels[pid] for pid in group) for group in (left, right)]
        mark_distance = sum(
            abs(Fraction(histograms[0][m], histograms[0].total()) - Fraction(histograms[1][m], histograms[1].total()))
            for m in MARKS
        )
        choices.append((
            sum(abs(difficulty[0][level] - difficulty[1][level]) for level in LEVELS),
            mark_distance,
            abs(histograms[0].total() - histograms[1].total()),
            left,
        ))
    chosen = min(choices)[-1]
    return {pid: "train" if pid in chosen else "validation" for pid in problems}


def assignment(levels: dict[str, str], grading: list[dict[str, str]], quarantined: frozenset[str]) -> dict[str, str]:
    """Assign every problem once: quarantined to heldout, graded problems balanced, the rest dealt by difficulty."""
    marks: dict[str, list[str]] = defaultdict(list)
    for row in grading:
        if row["Problem ID"] not in quarantined:
            marks[row["Problem ID"]].append(row["Points"])
    assigned = dict.fromkeys(quarantined, "heldout")
    available = SPLITS[:-1]
    graded = graded_split(marks, levels)
    assigned.update(graded)
    counts = Counter(graded.values())
    for level in LEVELS:
        problems = sorted(pid for pid in levels if pid not in assigned and levels[pid] == level)
        positions = [
            step if lap % 2 == 0 else len(available) - 1 - step
            for index in range(len(problems))
            for lap, step in (divmod(index, len(available)),)
        ]
        names = dict(zip(
            sorted(range(len(available)), key=lambda index: (-positions.count(index), index)),
            sorted(available, key=lambda name: (counts[name], available.index(name))),
            strict=True,
        ))
        for index, problem in enumerate(problems):
            assigned[problem] = names[positions[index]]
            counts[names[positions[index]]] += 1
    return assigned


def write(name: str, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    """Write one set under `build/` and check it against its pin."""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(columns))
    writer.writeheader()
    writer.writerows(rows)
    body = out.getvalue().encode("utf-8")
    if hashlib.sha256(body).hexdigest() != MEMBERS[name]:
        raise SystemExit(f"build/{name}.csv does not match its pinned sha256")
    (DATA / "build" / f"{name}.csv").write_bytes(body)


def main() -> None:
    proof = list(csv.DictReader(io.StringIO(fetched("proofbench.csv"))))
    grading = list(csv.DictReader(io.StringIO(filtered(fetched("gradingbench.csv")))))
    quarantined = frozenset(json.loads((DATA / "quarantine.json").read_bytes())["problems"])
    levels = {row["Problem ID"]: row["Level"] for row in proof}
    rows = joined(proof, grading)
    assigned = assignment(levels, rows, quarantined)
    writers = [
        {**{name: row[name] for name in PROOF_COLUMNS[1:-1]}, "Grading ID": row["Problem ID"],
         "Guidance": fenced(row, SHARED)}
        for row in proof
    ]
    (DATA / "build").mkdir(exist_ok=True)
    for split in SPLITS:
        members = [row for row in rows if assigned[row["Problem ID"]] == split]
        write(split, members, COLUMNS)
        if split != "heldout":
            write(f"{split}_positive", [row for row in members if float(row["Points"]) >= BAR], COLUMNS)
            write(f"{split}_negative", [row for row in members if float(row["Points"]) < BAR], COLUMNS)
        proofs = sorted((row for row in writers if assigned[row["Problem ID"]] == split), key=lambda row: row["Problem ID"])
        write(f"proof_{split}", proofs, PROOF_COLUMNS)
    print(f"wrote {len(MEMBERS)} sets to {DATA / 'build'}")


if __name__ == "__main__":
    main()
