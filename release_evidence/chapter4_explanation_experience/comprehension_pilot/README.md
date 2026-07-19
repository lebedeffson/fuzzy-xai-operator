# Comprehension pilot evidence

Status: `planned_not_run`.

This directory intentionally contains no participant responses. Copy
`response_template.csv` to a protected study workspace and record anonymized
responses from at least three domain specialists and three model integrators.
Each participant evaluates both `technical_baseline` and `human_explanation` in
counterbalanced order.

Score real responses with:

```bash
python scripts/score_comprehension_pilot.py responses.csv \
  --output release_evidence/explanation_experience/comprehension_pilot/summary.json
```

Do not commit names, contact details, free-text personal data, or fabricated rows.
The release tag remains blocked until the reviewed summary has `status=pass` and
the anonymized raw responses are archived under the approved research protocol.
