"""Raw-text object representation: evidence overlaid on the original text.

    python examples/03_text_explanation.py

Passing ``raw_object=<the original string>`` alongside its vectorized form
locates each measured feature contribution lexically inside the text
(word-boundary matching, never semantic/embedding-based) and renders it back
as highlighted HTML — or a tabular fallback for anything that isn't text.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DOCUMENTS = [
    "the router keeps dropping the wifi connection at night",
    "wifi signal is unstable and drops every evening",
    "invoice payment was declined by the billing system",
    "billing system rejected the invoice payment again",
]
LABELS = ["network", "network", "billing", "billing"]


def main() -> None:
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(DOCUMENTS).toarray()
    model = LogisticRegression(max_iter=2000).fit(features, LABELS)
    feature_names = list(vectorizer.get_feature_names_out())

    query_text = "wifi connection drops constantly every night, billing was never an issue"
    query_vector = vectorizer.transform([query_text]).toarray()[0]

    fx = FuzzyXAI.wrap(model, task="classification")
    result = fx.explain_one(
        query_vector,
        feature_names=feature_names,
        raw_object=query_text,  # <- the only addition versus example 01
    )

    repr_ = result.object_representation
    print("modality:", repr_["modality"])
    print("spans located in the raw text:", len(repr_["spans"]))
    print("features not found in the text (honestly disclosed, not guessed):", repr_["unmapped_features"])
    print()
    print(result.summary())

    result.visualize(view="object_representation", output="/tmp/fuzzyxai_example_03.png")
    print("\nWrote /tmp/fuzzyxai_example_03.png (highlighted text)")


if __name__ == "__main__":
    main()
