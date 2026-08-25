"""End-to-end demo: raw-text highlighting plus the optional SLM verbalizer.

Three scenarios, selected with a flag — run the first with zero setup:

    python examples/text_explanation_with_verbalizer.py                # deterministic, no network
    python examples/text_explanation_with_verbalizer.py --ollama        # local Ollama (see below)
    python examples/text_explanation_with_verbalizer.py --custom-backend  # your own backend class

Local Ollama setup is OS-specific — do not assume one command works
everywhere:
  Linux:   https://docs.ollama.com/linux    (install script / CLI package)
  macOS:   https://docs.ollama.com/macos    (desktop app)
  Windows: see https://ollama.com for the installer

Then, on any OS, from a terminal where the ``ollama`` command is available:

    ollama serve
    ollama pull qwen3:1.7b        # recommended default; qwen3:0.6b is smaller/weaker
    python examples/text_explanation_with_verbalizer.py --ollama

Not sure whether Ollama is reachable and the model is pulled? Run:

    python -m fuzzyxai.verbalization doctor

Without ``--ollama``/``--custom-backend`` (or if Ollama isn't reachable),
``result.verbalize()`` transparently falls back to the existing deterministic
summary — this script, and the library, never require Ollama to run.
"""

from __future__ import annotations

import argparse

from fuzzyxai import FuzzyXAI
from fuzzyxai.verbalization.backends import OllamaBackend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DOCUMENTS = [
    "the router keeps dropping the wifi connection at night",
    "wifi signal is unstable and drops every evening",
    "invoice payment was declined by the billing system",
    "billing system rejected the invoice payment again",
]
LABELS = ["network", "network", "billing", "billing"]


class EchoBackend:
    """Minimal example of a custom VerbalizationBackend (see verbalization/contracts.py).

    A real custom backend could call any local model, a different API, or a
    rule-based generator — the framework only depends on the ``generate()``
    method, never on a specific library. This one is deliberately trivial:
    it always picks the "structured" strict-mode connector so the demo shows
    a non-default path without needing any external service.
    """

    model = "echo-demo"

    def generate(self, prompt: str, *, response_schema=None) -> str:
        import json

        if response_schema and "order" in response_schema.get("properties", {}):
            # naive: pick the first claim_id mentioned in the prompt's DATA block
            import re

            claim_ids = re.findall(r'"claim_id":\s*"([^"]+)"', prompt)
            return json.dumps({"order": claim_ids, "connector": "structured"})
        return json.dumps({"sentences": []})  # this demo backend only supports strict mode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ollama", action="store_true", help="use a local Ollama backend for verbalize()")
    parser.add_argument("--custom-backend", action="store_true", help="use the EchoBackend example in this file")
    parser.add_argument("--model", default=None, help="Ollama model override (default: FUZZYXAI_OLLAMA_MODEL or qwen3:1.7b)")
    parser.add_argument("--mode", default="strict", choices=["strict", "rewrite"], help="verbalization mode (default: strict)")
    args = parser.parse_args()

    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(DOCUMENTS).toarray()
    model = LogisticRegression(max_iter=2000).fit(features, LABELS)
    feature_names = list(vectorizer.get_feature_names_out())

    query_text = "wifi connection drops constantly"
    query_vector = vectorizer.transform([query_text]).toarray()[0]

    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(
        query_vector,
        object_id="demo-0",
        feature_names=feature_names,
        raw_object=query_text,
    )

    print("=== Raw object with highlighted evidence (HTML) ===")
    print(result.object_representation["highlighted_html"])
    print()

    print("=== Deterministic summary (always available, no network) ===")
    print(result.summary())
    print()

    if args.ollama:
        backend = OllamaBackend(model=args.model)
    elif args.custom_backend:
        backend = EchoBackend()
    else:
        backend = None

    detailed = result.verbalize_detailed(backend=backend, audience="domain_user")
    print(f"=== verbalize_detailed() output (status={detailed.status}, backend={detailed.backend}, model={detailed.model}"
          f"{', fallback_reason=' + detailed.fallback_reason if detailed.fallback_reason else ''}) ===")
    print(detailed.text)
    print()
    print(f"guard_checks={detailed.guard_checks}  source_claim_ids={detailed.source_claim_ids}")


if __name__ == "__main__":
    main()
