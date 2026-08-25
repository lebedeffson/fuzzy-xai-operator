"""A real hand-written ANFIS-like fuzzy rule system, explained (P11).

    python examples/07_rule_based_model.py

FuzzyXAI's canonical API does not tie rule/fuzzy evidence to one specific
ANFIS library. Any model can supply it through a generic ``activated_rules``
channel — a plain list of dicts, each with a rule id, antecedent terms
(feature + linguistic term + real membership degree), an activation
strength, and a conclusion. Here that channel is populated by a small,
completely self-contained two-input Takagi-Sugeno-style fuzzy system: real
Gaussian membership functions, real product T-norm rule firing, real
weighted-average defuzzification — nothing about the evidence is invented,
and it is never dressed up as a linear feature contribution.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters.contracts_v2 import ExplanationContext, LocalModelEvidence
from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.adapters.model_v2 import ModelAdapterV2


def _gaussian_membership(x: float, mean: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2)


class SimpleANFIS:
    """A minimal two-input fuzzy rule system — not backed by any ANFIS library."""

    TERMS: ClassVar[dict[str, dict[str, tuple[float, float]]]] = {
        "temperature": {"low": (20.0, 8.0), "high": (35.0, 8.0)},
        "pressure": {"low": (1.0, 0.4), "high": (2.0, 0.4)},
    }
    RULES: ClassVar[list[tuple[str, dict[str, str], int]]] = [
        ("R1", {"temperature": "low", "pressure": "low"}, 0),
        ("R2", {"temperature": "low", "pressure": "high"}, 0),
        ("R3", {"temperature": "high", "pressure": "low"}, 1),
        ("R4", {"temperature": "high", "pressure": "high"}, 1),
    ]

    def _activations(self, temperature: float, pressure: float) -> list[dict[str, Any]]:
        values = {"temperature": temperature, "pressure": pressure}
        memberships = {
            feature: {term: _gaussian_membership(values[feature], mean, sigma) for term, (mean, sigma) in terms.items()}
            for feature, terms in self.TERMS.items()
        }
        activations = []
        for rule_id, antecedents, consequent in self.RULES:
            strength = 1.0  # product T-norm
            terms = []
            for feature, term in antecedents.items():
                degree = memberships[feature][term]
                strength *= degree
                terms.append({"feature": feature, "term": term, "membership_degree": degree, "feature_value": values[feature]})
            activations.append({"rule_id": rule_id, "terms": terms, "activation_strength": strength, "conclusion": str(consequent)})
        return activations

    def predict_one(self, temperature: float, pressure: float) -> tuple[int, list[dict[str, Any]]]:
        activations = self._activations(temperature, pressure)
        total_weight = sum(a["activation_strength"] for a in activations) or 1e-9
        weighted = sum(a["activation_strength"] * float(a["conclusion"]) for a in activations) / total_weight
        return round(weighted), activations

    def predict(self, X: Any) -> np.ndarray:
        rows = np.asarray(X)
        return np.array([self.predict_one(row[0], row[1])[0] for row in rows])


class ANFISLikeAdapter(ModelAdapterV2):
    """The generic fuzzy-rule contract: predict() plus an `activated_rules` channel."""

    adapter_id = "example_anfis_like_adapter"
    model_family = "fuzzy_rule_system"

    def predict(self, inputs: Any) -> ModelPrediction:
        rows = np.atleast_2d(np.asarray(inputs, dtype=float))
        predictions = self.model.predict(rows)
        return ModelPrediction(
            predictions=predictions.tolist(),
            probabilities=None,
            model_type="SimpleANFIS",
            adapter_id=self.adapter_id,
            metadata={"task_type": self.task_type.value},
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction, context
        rows = np.atleast_2d(np.asarray(inputs, dtype=float))
        _, activations = self.model.predict_one(rows[0][0], rows[0][1])
        return LocalModelEvidence(channels={"activated_rules": activations})


def main() -> None:
    model = SimpleANFIS()
    adapter = ANFISLikeAdapter(model, task="classification")
    fx = FuzzyXAI.wrap(model, adapter=adapter)

    temperature, pressure = 33.0, 1.9
    native_prediction, _ = model.predict_one(temperature, pressure)
    result = fx.explain_one([temperature, pressure], feature_names=["temperature", "pressure"])

    print("native prediction:", native_prediction)
    print("FuzzyXAI prediction:", result.prediction.predictions)
    print()

    rule_claims = [c for c in result.claims if c.claim_type == "fuzzy_rule"]
    print(f"{len(rule_claims)} fuzzy rule activation(s):")
    for claim in sorted(rule_claims, key=lambda c: -(c.strength or 0)):
        print(f"  {claim.subject_id}: activation={claim.strength:.2f}, effect={claim.effect}")

    print()
    print(result.summary())


if __name__ == "__main__":
    main()
