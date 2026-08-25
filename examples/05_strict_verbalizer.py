"""Optional SLM verbalization in strict mode (P6).

    python examples/05_strict_verbalizer.py

The SLM is never the source of the explanation — it only rephrases an
explanation already built from typed evidence. In *strict* mode a backend
may only choose which already-approved claim texts to use, their order, and
the connector between them; it cannot introduce a new fact, number, or
feature name. This script uses a small fake backend so it runs with zero
setup; see text_explanation_with_verbalizer.py for a real local-Ollama walkthrough.
"""

from __future__ import annotations

import json
import re

from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class OrderReversingStrictBackend:
    """A minimal strict backend: picks claim order and connector, invents nothing."""

    model = "example-strict-backend"

    def generate(self, prompt: str, *, response_schema=None) -> str:
        del response_schema
        claim_ids = re.findall(r'"claim_id":\s*"([^"]+)"', prompt)
        return json.dumps({"order": list(reversed(claim_ids)), "connector": "structured"})


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, task="classification")
    result = fx.explain_one(X_test[0])

    without_backend = result.verbalize_detailed()
    print(f"No backend  — status={without_backend.status!r} (deterministic, no network):")
    print(without_backend.text[:200])
    print()

    with_backend = result.verbalize_detailed(backend=OrderReversingStrictBackend())
    print(f"Strict backend — status={with_backend.status!r}, backend={with_backend.backend!r}:")
    print(with_backend.text[:400])
    print()
    print("source_claim_ids used:", with_backend.source_claim_ids)
    print("\nNote: rewrite mode exists too (surface guards on numbers/entities/")
    print("causal-certainty language) but its checks are surface consistency")
    print("checks, not proof that the whole rephrased text is semantically grounded.")


if __name__ == "__main__":
    main()
