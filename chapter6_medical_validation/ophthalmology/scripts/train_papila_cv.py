"""Train one pre-registered PAPILA outer-fold run (patient-level CV)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file, sha256_json
from chapter6_medical_validation.ophthalmology.src.datasets import configured_data_root, load_yaml
from chapter6_medical_validation.ophthalmology.src.models import build_classifier, model_fingerprint, save_checkpoint
from chapter6_medical_validation.ophthalmology.src.papila import papila_tensor

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class PapilaDataset:
    def __init__(self, rows: list[dict[str, str]], data_root: Path, raw_root: Path, contours: Path, cfg: dict[str, Any], *, training: bool, seed: int) -> None:
        self.rows, self.data_root, self.raw_root, self.contours, self.cfg = rows, data_root, raw_root, contours, cfg
        self.training, self.seed = training, seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        import torch
        row = self.rows[index]
        tensor = papila_tensor(self.data_root / row["image_path"], self.contours / f"{row['sample_id']}_disc_exp1.txt", self.cfg, training=self.training, seed=self.seed * 1_000_003 + index if self.training else None)
        return torch.from_numpy(tensor), int(row["diagnosis"]), row["sample_id"]


def _evaluate(model: Any, loader: Any, device: Any) -> tuple[float, list[dict[str, Any]]]:
    import torch
    from torch.nn import functional
    model.eval(); losses: list[float] = []; rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for inputs, labels, sample_ids in loader:
            logits = model(inputs.to(device)); probs = torch.softmax(logits, dim=1).cpu().numpy()
            losses.append(float(functional.cross_entropy(logits, labels.to(device)).item()))
            rows.extend({"sample_id": sid, "label": int(label), "probabilities": prob.tolist(), "logits": logit.tolist()} for sid, label, prob, logit in zip(sample_ids, labels, probs, logits.cpu().numpy(), strict=True))
    return float(np.mean(losses)), rows


def main() -> None:
    import torch
    from torch.nn import functional
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Train one PAPILA ResNet50 group-CV run")
    parser.add_argument("--data-root")
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    root = configured_data_root(args.data_root); verified = root / "eyes" / "papila" / "verified"
    split = json.loads((verified / "papila_cv_folds_seed2026.json").read_text(encoding="utf-8"))
    cfg = load_yaml(ROOT / "configs" / "model_resnet50_papila.yaml"); preprocessing = load_yaml(ROOT / "configs" / "preprocessing_papila.yaml")
    rows = _rows(verified / "papila_eye_labels.csv"); fold = split["folds"][str(args.fold)]
    by_patient = {name: set(fold[f"{name}_patient_ids"]) for name in ("train", "validation", "test")}
    if any(by_patient[left] & by_patient[right] for index, left in enumerate(by_patient) for right in list(by_patient)[index + 1:]): raise AssertionError("patient leakage")
    values = {name: [row for row in rows if row["patient_id"] in patients] for name, patients in by_patient.items()}
    if any(row["diagnosis"] not in {"0", "1"} for entries in values.values() for row in entries): raise AssertionError("suspect eye entered binary CV")
    raw_root = next((root / "eyes" / "papila" / "raw").glob("PapilaDB-PAPILA-*")); contours = raw_root / "ExpertsSegmentations" / "Contours"
    datasets = {name: PapilaDataset(entries, root, raw_root, contours, preprocessing, training=name == "train", seed=args.seed) for name, entries in values.items()}
    loaders = {name: DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=name == "train", num_workers=0, pin_memory=torch.cuda.is_available()) for name, dataset in datasets.items()}
    labels = np.asarray([int(row["diagnosis"]) for row in values["train"]]); counts = np.bincount(labels, minlength=2); weights = len(labels) / (2 * np.maximum(counts, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = build_classifier("resnet50", num_classes=2, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["optimizer"]["learning_rate"]), weight_decay=float(cfg["optimizer"]["weight_decay"])); criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    output = args.output or root / "eyes" / "papila" / "runs"; run_id = f"papila-resnet50-fold{args.fold}-seed{args.seed}-{split['manifest_sha256'][:12]}"; run_dir = output / run_id
    if run_dir.exists(): raise FileExistsError(f"frozen run already exists: {run_dir}")
    run_dir.mkdir(parents=True); history: list[dict[str, Any]] = []; best: tuple[float, int, str, str] | None = None; bad_epochs = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train(); train_losses: list[float] = []
        for inputs, targets, _ids in loaders["train"]:
            opt.zero_grad(set_to_none=True); logits = model(inputs.to(device)); loss = criterion(logits, targets.to(device)); loss.backward(); opt.step(); train_losses.append(float(loss.item()))
        validation_loss, validation_rows = _evaluate(model, loaders["validation"], device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss})
        if best is None or validation_loss < best[0]:
            checkpoint = save_checkpoint(model, run_dir / "best_model.pt", {"run_id": run_id, "fold": args.fold, "seed": args.seed, "epoch": epoch, "split_manifest_sha256": split["manifest_sha256"], "validation_loss": validation_loss})
            best = (validation_loss, epoch, checkpoint, model_fingerprint(model)); (run_dir / "validation_predictions.json").write_text(json.dumps({"rows": validation_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["early_stopping_patience"]): break
    assert best is not None
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_rows = _evaluate(model, loaders["test"], device)
    (run_dir / "test_predictions.json").write_text(json.dumps({"rows": test_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"run_id": run_id, "dataset": "PAPILA", "architecture": "resnet50", "fold": args.fold, "seed": args.seed, "primary_task": "healthy_vs_glaucoma", "roi_source": preprocessing["roi_source"], "model_fingerprint": best[3], "checkpoint_sha256": best[2], "best_epoch": best[1], "best_validation_loss": best[0], "test_loss": test_loss, "history": history, "split_manifest_sha256": split["manifest_sha256"], "preprocessing_sha256": sha256_json(preprocessing), "model_config_sha256": sha256_file(ROOT / "configs" / "model_resnet50_papila.yaml"), "patient_counts": fold["counts"]}
    (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
