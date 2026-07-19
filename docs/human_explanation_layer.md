# Human Explanation Layer

`HumanExplanation` is the verified communication layer above `ExplanationClaim` and `ExplanationGraph`. It does not invent a second explanation. It selects and groups evidence-backed claims around the questions a person needs to answer.

## First-level contract

The `domain_user` profile contains:

1. the model decision in domain language;
2. at most three main reasons;
3. at most two concerns;
4. a reliability statement;
5. one recommended action;
6. at most one tested change that can alter the result.

Internal rule IDs, subgroup IDs, claim IDs, E0-E5 labels, operator symbols, and raw action codes are forbidden in this first-level text. They remain available through `inspect()`, `audit()`, and technical audience profiles.

Every card contains non-empty `claim_refs` and `evidence_refs`. A renderer may hide those fields, but it must not remove them from the serialized result.

## Domain language

The framework cannot infer how a dataset column should be explained in a professional domain. Supply that contract through `ExplainPlan.domain_language`:

```python
from fuzzyxai import ExplainPlan

plan = ExplainPlan.default()
plan.domain_language = {
    "features": {
        "fracture_density": {
            "label": "трещиноватость породы",
            "meaning": "количество и плотность трещин в горном массиве",
            "high_text": "трещин больше, чем в большинстве исследованных участков",
            "low_text": "трещиноватость ниже типичного уровня",
        }
    },
    "classes": {"0": {"label": "низкий риск"}, "1": {"label": "повышенный риск"}},
    "actions": {
        "review": {
            "label": "проверить специалистом",
            "explanation": "данных недостаточно для автоматического решения",
        }
    },
}
```

Without this dictionary the framework uses conservative labels based on feature names and must not claim domain meaning it was not given.

## API

```python
result = FuzzyXAI.wrap(model, explain_plan=plan).explain_one(
    item,
    object_id="85",
    reference_data=X_train,
    reference_labels=y_train,
)

human = result.explain_for(audience="domain_user", language="ru")
print(human.decision.explanation)
print(human.main_reasons)
print(human.concerns)
print(human.reliability.explanation)
print(human.recommended_action.explanation)

print(result.summary(audience="domain_user", detail="short"))
print(result.summary(audience="ml_engineer", detail="full"))
result.inspect("claim:C004").provenance()
```

Supported audiences are `domain_user`, `ml_engineer`, `researcher`, and `auditor`. Compatibility aliases `user`, `expert`, and `audit` remain available for one release cycle.

## Ranking and comparison

Claims are ranked by decision importance, evidence status, domain relevance, and influence on the recommended action. Raw metric magnitude alone is not enough. Repeated claims are grouped. Comparative statements name their reference, such as percentile in the training reference or similarity representation.

Similarity text must state what was compared. For image masks, an IoU of 0.89 means 89% overlap of segmented regions. It is not a probability of the same diagnosis. Medical examples in this repository are controlled research-only fixtures.

## Evidence boundary

- Missing evidence produces an explicit limitation, never a fabricated value.
- Surrogate evidence remains labeled in engineer and audit profiles.
- Domain-user text translates technical values into effects before optional numbers.
- E0-E5 describes evidence depth; it is not a reliability or audience score.
- Demonstrated human comprehensibility remains an external gate until the documented pilot is run.
