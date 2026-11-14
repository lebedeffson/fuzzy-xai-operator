"""Reproducible end-to-end *demo* of the signed FuzzyXAI route contract.

Everything runs locally for demonstration; the signer and observer MUST be
separate trust domains in deployment. No clinical validation is implied.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from fuzzyxai.core.certified_route import (
    SignedPrediction,
    certify_json_document,
    sign_prediction,
)
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEMO_KEY = b"FuzzyXAI-DEMO-KEY-DO-NOT-USE-IN-PRODUCTION"


def payload(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def make_stage(anchor: SignedPrediction, index: int, *, event: str | None = None,
               score: float | None = None) -> dict:
    event = event or anchor.event_id
    score = anchor.score if score is None else score
    return {
        "terms": ["risk", "high"],
        "representation": {"type": "scalar", "value": score},
        "rules": [{"name": f"rule-{index}", "conditions": {"risk": "high"},
                   "conclusion": "review"}],
        "activations": {f"rule-{index}": 0.5},
        "uncertainty": 0.15,
        "trace": {
            "id": anchor.object_id, "version": anchor.model_version,
            "timestamp": "2026-11-14", "params": {}, "source": "sklearn-demo",
            "checksum": anchor.trace_checksum if index == 0 else f"derived-{index}",
        },
        "metadata": {
            "semantic_contract": {
                "object_id": anchor.object_id, "event_id": event,
                "model_version": anchor.model_version, "unit": anchor.unit,
                "source_sha256": anchor.source_sha256,
                "model_sha256": anchor.model_sha256,
            },
            "risk": score,
        },
    }


def run(output_dir: Path) -> dict:
    dataset = load_breast_cancer()
    train, test = train_test_split(np.arange(len(dataset.data)), random_state=17,
                                   test_size=0.2, stratify=dataset.target)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=17))
    model.fit(dataset.data[train], dataset.target[train])
    index = int(test[0])
    prediction = model.predict_proba(dataset.data[[index]])[0]
    p_malignant = float(prediction[0])  # sklearn target 0 = malignant
    pipeline_state = {
        "scaler_mean": model.named_steps["standardscaler"].mean_.tolist(),
        "scaler_scale": model.named_steps["standardscaler"].scale_.tolist(),
        "coefficients": model.named_steps["logisticregression"].coef_.tolist(),
        "intercept": model.named_steps["logisticregression"].intercept_.tolist(),
        "classes": model.named_steps["logisticregression"].classes_.tolist(),
    }
    model_hash = sha256(payload(pipeline_state)).hexdigest()
    source_hash = sha256(payload({"features": dataset.data[index].tolist(),
                                   "ordered_names": list(dataset.feature_names)})).hexdigest()
    # This is the *producer* step. In production it must be isolated from the observer.
    receipt = sign_prediction(SignedPrediction(
        object_id=f"breast_cancer:{index}", event_id="malignant",
        model_version="logistic-17-"+model_hash[:12],
        unit="probability", score=p_malignant,
        source_sha256=source_hash, model_sha256=model_hash,
        trace_checksum=sha256(payload({"object_index": index,
                                        "score": p_malignant})).hexdigest(),
    ), secret=DEMO_KEY)
    beta = {"repr": 0.2, "rules": 0.2, "activations": 0.2,
            "uncertainty": 0.15, "trace": 0.15, "reduction": 0.1}
    stage0, stage1, stage2 = (make_stage(receipt, i) for i in range(3))
    base = {"schema": "fuzzyxai-certified-route-v1",
            "signed_prediction": asdict(receipt),
            "policy": {"beta": beta, "gamma_limit": 0.9},
            "stages": [stage0, stage1, stage2]}
    import copy
    clean = copy.deepcopy(base)
    tampered = copy.deepcopy(base)
    for stage in tampered["stages"]:
        stage["metadata"]["risk"] = 1.0 - p_malignant
        stage["representation"]["value"] = 1.0 - p_malignant
        stage["metadata"]["semantic_contract"]["event_id"] = "benign"
    missing = copy.deepcopy(base)
    missing["stages"][2]["metadata"].pop("semantic_contract")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, case in (("clean", clean), ("coherent_semantic_spoof", tampered),
                       ("missing_witness", missing)):
        (output_dir / f"{name}.json").write_text(json.dumps(case, indent=2), encoding="utf8")
        cert = certify_json_document(case, verification_secret=DEMO_KEY).to_dict()
        result[name] = cert
    (output_dir / "certificates.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    assert [result[x]["status"] for x in ("clean", "coherent_semantic_spoof", "missing_witness")] == [
        "certified", "rejected", "insufficient_evidence",
    ]
    return {"trained_model": "sklearn LogisticRegression with StandardScaler",
            "test_object_id": receipt.object_id, "model_sha256": model_hash,
            "malignant_probability": p_malignant,
            "certificates": {name: cert["status"] for name, cert in result.items()},
            "output": str(output_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("./certified_route_demo_output"))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), ensure_ascii=False, indent=2))