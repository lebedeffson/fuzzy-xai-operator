# FuzzyXAI blinded expert review form

Status: `prepared_not_run`

For each item, record the following without viewing the model's true label:

| Field | Allowed value |
|---|---|
| reviewer_id | pseudonymous reviewer code |
| review_item_id | ID from the blinded packet |
| recommended_action | `accept`, `review`, or `block` |
| explanation_sufficient | `yes` or `no` |
| main_reason_understood | `yes` or `no` |
| concern_understood | `yes` or `no` |
| confidence_1_to_5 | integer 1-5 |
| notes | optional free text without personal data |

The form measures technical decision support. It is not a clinical validation
instrument and must not be presented as one.
