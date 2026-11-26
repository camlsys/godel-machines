# `imo` seats: designed behaviour and the test that pins it

Each row is a behaviour the seats in this tree were designed to have, with the test in
`test_imo_seats.py` that fails when it stops being true. Run from the node root with
`python -m pytest tests/seats`.

| designed behaviour | test |
|---|---|
| The reply is the proof; it is filed whole to `marking`, with no restated problem or heading | `test_the_writer_files_the_reply_whole_and_sends_the_problem_first` |
| The prompt puts the problem first and carries its LaTeX byte for byte | `test_the_writer_files_the_reply_whole_and_sends_the_problem_first` |
| One consultation of the node's grader on the first draft; the revision prompt carries problem, draft, mark, reasoning; the revision is filed; the review is recorded as evidence | `test_the_writer_asks_the_grader_once_and_files_the_revision` |
| A reviewer that raises, or a tree that seats none, costs the revision and never the proof | `test_a_broken_reviewer_costs_the_revision_and_never_the_proof` |
| `REVIEW_ROUNDS` is the round count, one by default; zero is the one-call seat | `test_zero_rounds_is_the_one_call_seat`, `test_the_default_is_one_round_and_is_stated_on_the_module` |
| No grant, no call | `test_the_writer_refuses_without_a_grant` |
| The mark is the one integer on 0..7 in the reply's JSON block, read through `shared.reply`; no block, two different marks, a string, float or boolean, or a value off the scale is `MALFORMED` (-1), which is off the scale on purpose | `test_the_grader_reads_every_reply_shape` |
| The prompt puts the work first and asks for the block once | `test_the_grader_prompt_puts_the_work_first_and_asks_for_the_block_once` |
| Blindness to provenance: a judged proof is joined to the row's own guidance under `### Response`, the layout a corpus row has, and the seat is never told which it holds | `test_the_grader_marks_a_judged_proof_in_the_corpus_layout`, `test_the_grader_marks_a_corpus_row_as_staged` |
| The grade is `{"score": n}` in `grade.txt` on both arms, and both episodes declare that channel; the runtime reads the verdict off the declared channel and refuses a job declaring other than one | `test_the_grader_marks_a_judged_proof_in_the_corpus_layout` |
| An unreadable reply is recorded as malformed with `parsed.malformed: true`, never as a zero | `test_the_grader_records_an_unreadable_reply_as_malformed_not_zero` |
| `review(problem, proof, chat)` returns the mark and the whole reply, the draft staged last before the instruction | `test_the_grader_reviews_a_draft_and_returns_mark_and_reasoning` |
| No `marking` staged, or no grant: refuse by name | `test_the_grader_refuses_no_work_and_no_grant` |
| Every evidence key a seat states is one its arm declares (`reply`, `parsed` for the grader; `reply`, `staged`, `review` for the writer) | `test_every_evidence_a_seat_states_is_declared_by_its_arm` |
