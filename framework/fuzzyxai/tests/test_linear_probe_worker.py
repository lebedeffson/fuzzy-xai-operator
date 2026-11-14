"""Cross-process, trained-model verification of the linear HMAC probe worker."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import asdict, replace

import numpy as np
import pytest
from fuzzyxai.core.certified_route import (
    RouteStatus,
    SignedLocalEffectWitness,
    SignedPrediction,
    certify_route_with_attested_local_effects,
    sign_prediction,
)
from fuzzyxai.core.explanation_object import ExplanationObject, Rule, Trace
from fuzzyxai.selective_observer.evidence_identity import LocalExplanationEvidence
from fuzzyxai.selective_observer.local_effects import (
    SUPPORTED_PROTOCOL,
    LocalEffectEvidence,
    hash_numeric_input,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE_KEY = b"test-only-origin-source-key-secure-32"
EFFECT_KEY = b"test-only-separate-probe-key-secure-32"
PRODUCER_ID = "separately-invoked-logit-probe"
BETA = {
    "repr": 0.2, "rules": 0.2, "activations": 0.2,
    "uncertainty": 0.15, "trace": 0.15, "reduction": 0.1,
}


class _Scalar:
    def __init__(self, value):
        self.value = value

    def distance(self, other):
        return abs(self.value - other.value)


def _stage(origin, number):
    return ExplanationObject(
        terms={"yes", "no"}, representation=_Scalar(origin.score),
        rules=[Rule(f"rule-{number}", {"review": "required"}, "inspect")],
        activations={f"rule-{number}": 0.7}, uncertainty=0.11,
        trace=Trace(origin.object_id, origin.model_version, "2026-11-14",
                    source="origin-probe", checksum=(origin.trace_checksum if number == 0 else f"stage-{number}")),
        metadata={
            "risk": origin.score,
            "semantic_contract": {
                "object_id": origin.object_id, "event_id": origin.event_id,
                "model_version": origin.model_version, "unit": origin.unit,
                "source_sha256": origin.source_sha256,
                "model_sha256": origin.model_sha256,
            },
        },
    )


def _serializable_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


@pytest.fixture()
def signed_fixture(tmp_path):
    x_train = np.asarray([
        [-2.0, -1.0], [-1.0, 0.0], [0.0, -1.0],
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.5, 1.5],
        [1.4, 0.9], [-0.9, -1.5],
    ])
    y_train = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0])
    fitted = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=500, random_state=17),
    ).fit(x_train, y_train)
    scaler = fitted.named_steps["standardscaler"]
    classifier = fitted.named_steps["logisticregression"]
    names = ("feature_0", "feature_1")
    model_version = "trained-isolated-1"
    manifest = {
        "schema": "fuzzyxai-linear-binary-logit-model-v1",
        "model_version": model_version, "target_class": "positive",
        "producer_id": PRODUCER_ID, "feature_names": list(names),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "classes": [0, 1],
    }
    filepath = tmp_path / "model.json"
    filepath.write_text(json.dumps(manifest), encoding="utf-8")
    model_hash = _serializable_sha({
        "coef": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    })
    keyfile = tmp_path / "probe.key"
    keyfile.write_bytes(EFFECT_KEY)
    keyfile.chmod(stat.S_IRUSR | stat.S_IWUSR)
    sample = (1.25, 0.8)
    reference = tuple(float(v) for v in scaler.mean_)
    origin = sign_prediction(
        SignedPrediction(
            object_id="inference-item-1", event_id="positive",
            model_version=model_version, unit="probability",
            score=float(fitted.predict_proba([sample])[0, 1]),
            source_sha256=hash_numeric_input(names, sample),
            model_sha256=model_hash,
            trace_checksum=hashlib.sha256(b"measured-inference").hexdigest(),
            complement_event_id="negative",
        ), secret=SOURCE_KEY,
    )
    request = {
        "schema": "fuzzyxai-linear-response-request-v1",
        "source_prediction": asdict(origin),
        "input_values": list(sample), "reference_values": list(reference),
    }
    contributions = tuple(
        float((actual - base) / scale * weight)
        for actual, base, scale, weight in zip(
            sample, reference, scaler.scale_, classifier.coef_[0], strict=True,
        )
    )
    local = LocalExplanationEvidence(
        object_id=origin.object_id, model_version=model_version,
        target_class="positive", scope="local", feature_names=names,
        contributions=contributions,
        source_sha256=hashlib.sha256(b"independent-local-attribution").hexdigest(),
        input_sha256=origin.source_sha256,
        reference_sha256=hash_numeric_input(names, reference),
        attribution_protocol=SUPPORTED_PROTOCOL, score_channel="target_logit",
    )
    return filepath, keyfile, request, origin, fitted, local


def _run_worker(manifest, keyfile, request, *, key_env=True):
    env = dict(os.environ)
    env.pop("FUZZYXAI_PROBE_KEY_FILE", None)
    env.pop("FUZZYXAI_REFERENCE_KEY", None)
    if key_env:
        env["FUZZYXAI_PROBE_KEY_FILE"] = str(keyfile)
    proc = subprocess.run(
        [sys.executable, "-m", "fuzzyxai.core.linear_probe_worker",
         "--manifest", str(manifest)],
        input=json.dumps(request), encoding="utf-8",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, check=False, timeout=15,
    )
    return proc.returncode, json.loads(proc.stdout), proc.stdout


def test_worker_produces_own_model_response_and_separate_signed_certificate(signed_fixture):
    manifest, keyfile, req, origin, fitted, local = signed_fixture
    code, response, raw = _run_worker(manifest, keyfile, req)
    assert code == 0
    assert response["status"] == "signed_model_response"
    assert SOURCE_KEY.decode() not in raw and EFFECT_KEY.decode() not in raw
    assert response["model_sha256"] == origin.model_sha256
    effects_dict = response["model_effects"]
    effects = LocalEffectEvidence(**{
        **effects_dict,
        "feature_names": tuple(effects_dict["feature_names"]),
        "effect_on_score": tuple(effects_dict["effect_on_score"]),
    })
    witness = SignedLocalEffectWitness(**response["signed_effect_witness"])
    assert effects.effect_on_score == pytest.approx(local.contributions, abs=1e-9)
    assert effects.explained_score == pytest.approx(
        float(fitted.decision_function([req["input_values"]])[0]), abs=1e-9,
    )
    args = {
        "stages": [_stage(origin, 0), _stage(origin, 1)],
        "signed_prediction": origin, "verification_secret": SOURCE_KEY,
        "beta": BETA, "local_evidence": local, "source_model_effects": effects,
        "signed_effect_witness": witness,
        "effects_verification_secret": EFFECT_KEY,
        "expected_effect_producer_id": PRODUCER_ID,
    }
    assert certify_route_with_attested_local_effects(**args).status is RouteStatus.CERTIFIED
    assert certify_route_with_attested_local_effects(
        **{**args, "source_model_effects": replace(effects, effect_on_score=(0.8, 0.1))}
    ).status is RouteStatus.REJECTED
    assert certify_route_with_attested_local_effects(
        **{**args, "expected_effect_producer_id": "untrusted-prober"}
    ).status is RouteStatus.REJECTED
    assert certify_route_with_attested_local_effects(
        **{**args, "effects_verification_secret": SOURCE_KEY}
    ).status is RouteStatus.INSUFFICIENT


@pytest.mark.parametrize("mutation", [
    "model_sha", "input_hash", "input_value", "signed_probability",
    "model_version", "wrong_class", "unknown_extra_field", "nonfinite_input",
    "unapproved_reference",
])
def test_worker_fails_closed_for_model_and_source_mismatches(signed_fixture, mutation):
    manifest, keyfile, request, origin, _fitted, _local = signed_fixture
    request = json.loads(json.dumps(request))
    if mutation == "model_sha":
        request["source_prediction"]["model_sha256"] = "f" * 64
    elif mutation == "input_hash":
        request["source_prediction"]["source_sha256"] = "e" * 64
    elif mutation == "input_value":
        request["input_values"][0] += 0.3
    elif mutation == "signed_probability":
        request["source_prediction"]["score"] = max(0.0, origin.score - 0.2)
    elif mutation == "model_version":
        request["source_prediction"]["model_version"] = "unrecognized"
    elif mutation == "wrong_class":
        request["source_prediction"]["event_id"] = "negative"
    elif mutation == "unknown_extra_field":
        request["model_effects"] = {"effect_on_score": [0.0, 0.0]}
    elif mutation == "nonfinite_input":
        request["input_values"][0] = float("nan")
    elif mutation == "unapproved_reference":
        request["reference_values"][0] += 0.15
    code, response, raw = _run_worker(manifest, keyfile, request)
    assert code == 3
    assert response["status"] == "insufficient_evidence"
    assert "signed_effect_witness" not in response
    assert EFFECT_KEY.decode() not in raw


def test_worker_refuses_missing_or_open_signing_key(signed_fixture):
    manifest, keyfile, request, _origin, _fitted, _local = signed_fixture
    assert _run_worker(manifest, keyfile, request, key_env=False)[0] == 3
    keyfile.chmod(0o644)
    if os.name == "posix":
        assert _run_worker(manifest, keyfile, request)[0] == 3


def test_worker_stream_handles_multiple_responses_without_signing_invalid_input(signed_fixture):
    manifest, keyfile, request, _origin, _fitted, _local = signed_fixture
    changed = json.loads(json.dumps(request))
    changed["reference_values"][0] += 0.2
    requests = (request, changed, request)
    env = dict(os.environ)
    env["FUZZYXAI_PROBE_KEY_FILE"] = str(keyfile)
    env.pop("FUZZYXAI_REFERENCE_KEY", None)
    result = subprocess.run(
        [sys.executable, "-m", "fuzzyxai.core.linear_probe_worker",
         "--manifest", str(manifest), "--stream", "--max-requests", "3"],
        input="".join(json.dumps(item) + "\n" for item in requests),
        text=True, capture_output=True, env=env, check=False, timeout=15,
    )
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0
    assert len(records) == 3
    assert [item["status"] for item in records] == [
        "signed_model_response", "insufficient_evidence", "signed_model_response",
    ]
    assert records[0] == records[2]
    assert "signed_effect_witness" not in records[1]
    assert SOURCE_KEY.decode() not in result.stdout and EFFECT_KEY.decode() not in result.stdout
