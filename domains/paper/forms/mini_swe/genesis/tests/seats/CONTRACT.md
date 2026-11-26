# `paper` seats: designed behaviour and the test that pins it

Each row is a behaviour the seats in this tree were designed to have, with the test in
`test_paper_seats.py` that fails when it stops being true. Run from the node root with
`python -m pytest tests/seats`.

| designed behaviour | test |
|---|---|
| The writer is staged the blind channel `input` (title and abstract, by position) and never the body or the decision; the reviewer is staged `paper_text` by name | `test_the_writer_is_staged_the_blind_channel_and_the_reviewer_the_body` |
| The writer sends the premise first, then its instruction, once, in one model call | `test_the_writer_sends_the_premise_first_and_its_instruction_once` |
| The model's whole reply is the draft; nothing is parsed out of it, so a fence or LaTeX cannot lose it | `test_the_writer_writes_the_reply_whole_as_the_draft` |
| An empty premise (no files, a file of no bytes, two of no bytes, whitespace) refuses before the call is paid for | `test_the_writer_refuses_an_empty_premise_before_spending_the_call` |
| The reviewer sends the review form that names the `Decision` field it parses first, then the whole paper, once | `test_the_reviewer_sends_its_instruction_first_and_the_whole_paper_once` |
| The reviewer writes `verdict.txt` as `{"score": x}` and states `reply`, `reviewed`, `parsed` as evidence | `test_the_reviewer_writes_a_numeric_verdict_and_the_three_evidence_files` |
| The verdict vocabulary: one `Decision` of `Accept` (1.0) or `Reject` (0.0), bare, fenced or behind prose; none, several, off-vocabulary, unparsable or nested is `MALFORMED` (-1.0), which is no decision's value | `test_the_reviewer_reads_one_decision_out_of_each_reply_shape` |
| The reviewer runs on two arms, generate (a real paper) and judge (the writer's draft), with one vocabulary, so it cannot tell which | `test_the_reviewer_has_one_vocabulary_on_both_of_its_arms` |
| The writer's `paper_text` is the channel the reviewer reads | `test_the_writer_draft_is_the_channel_the_reviewer_reads` |
| The meta agent's self-review is this reviewer reading the work so far under `### Draft under review` | `test_self_review_is_the_reviewer_reading_a_draft_in_progress` |
| Each seat answers within the arm its form declares: the writer's evidence is `reply`, `staged`; the reviewer's `parsed`, `reply`, `reviewed` on both arms | `test_each_seat_answers_within_the_arm_its_form_declares` |
| The rows a node may study are `train_positive` (50), `train_negative` (50), `write_train` (100); scores come off `validation_positive`, `validation_negative`, `write_validation`, never staged; `heldout` is named nowhere | `test_the_rows_a_node_may_study_are_the_train_halves_and_never_the_measured_ones` |
