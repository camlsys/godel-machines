# Copyright 2026 Alex Iacob
# SPDX-License-Identifier: Apache-2.0
"""Build the `paper` corpus under `domains/paper/data/build/` from the hyperagents paper-review dataset.

Downloads `dataset.csv` from facebookresearch/hyperagents at a pinned commit and checks its sha256.
`split.json` names the 100 papers of each split (50 accepted, 50 rejected). Each row carries the
paper's abstract (the writer's input: the text up to the first section heading, or up to the
marker below where the dataset has no heading after the abstract), its full text (the reviewer's
input) and the hidden decision. Every written member is checked against its pinned sha256.

    python tools/operations/build_paper_corpus.py
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "domains" / "paper" / "data"
UPSTREAM = (
    "https://raw.githubusercontent.com/facebookresearch/hyperagents/"
    "9d080848e46b6627357846824d4a9febb5202ed1/domains/paper_review/dataset.csv"
)
UPSTREAM_SHA256 = "efa8e77721d6d41c5d9ea69aaab8f0c16664618876e2c9dfe897a3852bb242c9"
MEMBERS = {
    "heldout": "ede35caa9553cbc957998d653b4ed7e8f66b3a734eeb01f838d02559a798f7ea",
    "train": "4b1eca0083d42a18e8a391435cbda720b1481995d0e5f046c9c4c86f7f1c1ed7",
    "train_negative": "ff838cd068e64883f06cd4214e09be0884c44cc1ace06169085dd2d60b8e305c",
    "train_positive": "23af04cccabaa8cb061cf61b35214e95252e08767c5674dab2d2377228c17759",
    "validation": "02b4c0831190822c861a94d893dd295f40858ba92684a3a54203880c7b56506e",
    "validation_negative": "2963060961076c99c21343116b1bf1e4bf40eb48c55a62801b6594d2a61e7f10",
    "validation_positive": "1807218d1034d730affa6bda87e5d1de6f8b0099fb0a154ea40465be05a0d5d7",
    "write_train": "4b1eca0083d42a18e8a391435cbda720b1481995d0e5f046c9c4c86f7f1c1ed7",
    "write_validation": "02b4c0831190822c861a94d893dd295f40858ba92684a3a54203880c7b56506e",
}
COLUMNS = ("question_id", "abstract", "paper_text", "decision")
SECTION = re.compile(r"\\n\\nSection:(?:[ \t]|\\n)*[A-Za-z0-9]")
#: Papers whose abstract runs into body text: the abstract ends after this phrase.
MARKERS = {
    "fMTPkDEhLQ": "in this general setting.",
    "xDrFWUmCne": "https://github.com/vinhsuhi/LD3.",
    "UvTo3tVBk2": "showing promise on code and math tasks.",
    "APGXBNkt6h": "Our code is open-sourced 1 .",
    "gGl0n7Onug": "fine-tuning steps on influential examples.",
    "d47iuwOt3j": "validate the effectiveness, efficiency and security of our proposed method.",
    "EytBpUGB1Z": "compressing the KV (Key-Value) cache.",
    "5Gw9YkJkFF": "demonstrating the effectiveness of our techniques.",
    "x4W8P7ybTE": "demonstrating strong performance in single and multi-turn medical conversations.",
    "zqXANcFO9T": "particularly in scenarios involving high compression and significant data heterogeneity.",
    "Q6a9W6kzv5": "bridging the gap between VLMs and physical world understanding.",
    "NbbsRnPBoS": "dispelling some of the past negative results on the role of depth.",
    "EzjsoomYEb": "Code and data are available at https://github.com/yoavgelberg/SMCN.",
    "kUuKFW7DIF": "Multilingual SUPERB (ML-SUPERB).",
    "vyFSyfiOIu": "compared to previous SOTA multimodal methods 1 .",
    "3j2nasmKkP": "Code is available at https: //github.com/LUMIA-Group/Cluster-wise-Graph-Transformer.",
    "MFWgLCWgUB": "O(log 2 k) on the competitive ratio of this algorithm.",
    "EWKPEtwjTy": "comparable results across various other control tasks.",
    "8qu52Fl1Dt": "an 81% improvement in spatiotemporal metrics.",
    "4G2DN4Kjk1": "(4) reducing the number of batches and allowing smaller batch sizes.",
    "OZ7aImD4uQ": "directly optimize the mechanistic interpretability of models.",
    "gU58d5QeGv": "work motivates more emphasis on the prioritization of both performance and computational accessibility.",
    "uCZI8gSfD4": "all within less or equivalent pre-training compute budgets.",
    "hCdqDkA25J": "reproducibility-convergence trade-off in the context of convex optimization.",
    "GrMczQGTlA": "generative modeling of sensorimotor sequences.",
    "h922Qhkmx1": "a step toward general audio models.",
    "473sH8qki8": "further guide the training of an observation-based policy in the target environment.",
    "yfkvUJEY6i": "meaningfully shifts which patients are considered high-risk.",
    "j5BuTrEj35": "freely available at https://github.com/huggingface/datablations.",
    "zuuhtmK1Ub": "a promising tool for geoscience and other physics-based applications.",
    "fKwG6grp8o": "corrections to both the bias and variance of network dynamics due to finite width.",
    "EQAHilKZ8D": "demonstrated the substantial effectiveness of our method.",
    "vpJMJerXHU": "https://github.com/luodhhh/ModernTCN.",
    "mQPNcBWjGc": "Code and checkpoints are available on GitHub.",
    "Zsfiqpft6K": "available at https://ku-cvlab.github.io/DiffMatch.",
    "Oo7HY9kmK6": "exponential dependence that would result from prior analyses).",
    "ywgwArtbDq": "machines have not caught up with humans -yet.",
    "RWJX5F5I9g": "general, brain-inspired algorithm for enhancing exploration in RL.",
    "GGlpykXDCa": "facilitate LLMs' multi-table capabilities in real-world scenarios.",
    "QQBPWtvtcn": "even with reduced computational resources (1-2 GPUs).",
}

csv.field_size_limit(sys.maxsize)


def abstract(text: str, *, key: str) -> str:
    """Return the quoted title and abstract that open a paper's quoted text."""
    prefix = text[: SECTION.search(text).start()]
    quote = prefix[0] if prefix[0] in "\"'" else ""
    body = prefix[1:] if quote else prefix
    if (marker := MARKERS.get(key)) is not None:
        marker = marker.replace(quote, f"\\{quote}") if quote else marker
        body = body[: body.index(marker) + len(marker)]
    return f"{quote}{body.rstrip()}{quote}"


def unquoted(value: str) -> str:
    """Return a column stored as a Python string literal as the text it stands for."""
    if len(value) < 2 or value[0] not in "\"'" or value[-1] != value[0]:
        return value
    try:
        literal = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value
    return literal if isinstance(literal, str) else value


def write(name: str, rows: list[dict[str, str]]) -> None:
    """Write one set under `build/` and check it against its pin."""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    body = out.getvalue().encode("utf-8")
    if hashlib.sha256(body).hexdigest() != MEMBERS[name]:
        raise SystemExit(f"build/{name}.csv does not match its pinned sha256")
    (DATA / "build" / f"{name}.csv").write_bytes(body)


def main() -> None:
    with urllib.request.urlopen(UPSTREAM, timeout=300) as response:
        raw = response.read()
    if (seen := hashlib.sha256(raw).hexdigest()) != UPSTREAM_SHA256:
        raise SystemExit(f"{UPSTREAM} has sha256 {seen}, expected {UPSTREAM_SHA256}")
    papers = {row["question_id"]: row for row in csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))}
    (DATA / "build").mkdir(exist_ok=True)
    for split, keys in json.loads((DATA / "split.json").read_bytes()).items():
        rows = [
            {
                "question_id": key,
                "abstract": unquoted(abstract(papers[key]["paper_text"], key=key)),
                "paper_text": unquoted(papers[key]["paper_text"]),
                "decision": papers[key]["outcome"],
            }
            for key in keys
        ]
        write(split, rows)
        if split != "heldout":
            write(f"{split}_positive", [row for row in rows if row["decision"] == "accept"])
            write(f"{split}_negative", [row for row in rows if row["decision"] != "accept"])
            write(f"write_{split}", rows)
    print(f"wrote {len(MEMBERS)} sets to {DATA / 'build'}")


if __name__ == "__main__":
    main()
