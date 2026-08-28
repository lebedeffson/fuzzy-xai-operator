from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from chapter6_medical_validation.ophthalmology.src.artifact_io import sha256_file, sha256_json
from chapter6_medical_validation.ophthalmology.src.datasets import configured_data_root, load_yaml, read_frozen_split
from chapter6_medical_validation.ophthalmology.src.metrics import classification_metrics
from chapter6_medical_validation.ophthalmology.src.models import build_classifier, model_fingerprint, save_checkpoint
from chapter6_medical_validation.ophthalmology.src.preprocessing import preprocess_image

ROOT = Path(__file__).resolve().parents[1]


class EyeDataset:
    def __init__(self, rows: list[dict[str, Any]], data_root: Path, preprocess_cfg: dict[str, Any], split: str, seed: int):
        self.rows, self.data_root, self.preprocess_cfg, self.split, self.seed = rows, data_root, preprocess_cfg, split, seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        import torch

        row = self.rows[index]
        item = preprocess_image(
            self.data_root / row["image_path"],
            self.preprocess_cfg,
            split=self.split,
            seed=self.seed * 1_000_003 + index if self.split == "train" else None,
        )
        return torch.from_numpy(item.normalized_chw), int(row["label"]), str(row["sample_id"])


def _evaluate(model: Any, loader: Any, device: Any) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    import torch
    from torch.nn import functional

    model.eval()
    losses: list[float] = []
    truth: list[int] = []
    probabilities: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for inputs, labels, sample_ids in loader:
            logits = model(inputs.to(device))
            losses.append(float(functional.cross_entropy(logits, labels.to(device)).item()))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            truth.extend(labels.numpy().tolist())
            probabilities.extend(probs.tolist())
            rows.extend(
                {"sample_id": sid, "label": int(label), "logits": logit, "probabilities": prob}
                for sid, label, logit, prob in zip(sample_ids, labels, logits.cpu().numpy().tolist(), probs.tolist(), strict=True)
            )
    return float(np.mean(losses)), classification_metrics(np.asarray(truth), np.asarray(probabilities)), rows


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Train registered CH6 five-grade classifier")
    parser.add_argument("--architecture", choices=["vgg16", "efficientnet_b0"], default="vgg16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--data-root")
    parser.add_argument("--split", type=Path, default=ROOT / "outputs" / "manifests" / "aptos_split.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "runs")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_cfg = load_yaml(ROOT / "configs" / f"model_{args.architecture.replace('_', '')}.yaml")
    preprocess_cfg = load_yaml(ROOT / "configs" / "preprocessing_eye.yaml")
    split = read_frozen_split(args.split)
    data_root = configured_data_root(args.data_root)
    datasets = {
        "train": EyeDataset(split["records"]["train"], data_root, preprocess_cfg, "train", args.seed),
        "validation": EyeDataset(split["records"]["validation"], data_root, preprocess_cfg, "validation", args.seed),
    }
    loaders = {
        name: DataLoader(value, batch_size=int(model_cfg["batch_size"]), shuffle=name == "train", num_workers=0)
        for name, value in datasets.items()
    }
    labels = np.asarray([row["label"] for row in split["records"]["train"]], dtype=int)
    counts = np.bincount(labels, minlength=5)
    weights = len(labels) / (5.0 * np.maximum(counts, 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier(args.architecture, num_classes=5, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(model_cfg["optimizer"]["learning_rate"]), weight_decay=float(model_cfg["optimizer"]["weight_decay"]))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    run_id = f"{model_cfg['model_id']}-seed-{args.seed}-{split['manifest_sha256'][:12]}"
    run_dir = args.output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    history, best = [], None
    for epoch in range(1, int(model_cfg["epochs"]) + 1):
        model.train()
        train_losses = []
        for inputs, targets, _sample_ids in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        val_loss, val_metrics, val_rows = _evaluate(model, loaders["validation"], device)
        epoch_row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss, "validation_qwk": val_metrics["quadratic_weighted_kappa"]}
        history.append(epoch_row)
        score = (val_loss, -float(val_metrics["quadratic_weighted_kappa"]))
        if best is None or score < best[0]:
            metadata = {"run_id": run_id, "epoch": epoch, "seed": args.seed, "model_config_sha256": sha256_json(model_cfg), "preprocessing_config_sha256": sha256_json(preprocess_cfg), "split_manifest_sha256": split["manifest_sha256"], "validation_metrics": val_metrics}
            checkpoint_sha = save_checkpoint(model, run_dir / "best_model.pt", metadata)
            best = (score, epoch, checkpoint_sha, val_metrics, val_rows, model_fingerprint(model))
    assert best is not None
    payload = {"run_id": run_id, "architecture": args.architecture, "seed": args.seed, "history": history, "best_epoch": best[1], "checkpoint_sha256": best[2], "model_fingerprint": best[5], "split_manifest_sha256": split["manifest_sha256"], "model_config_sha256": sha256_file(ROOT / "configs" / f"model_{args.architecture.replace('_', '')}.yaml"), "preprocessing_config_sha256": sha256_file(ROOT / "configs" / "preprocessing_eye.yaml"), "validation_metrics": best[3]}
    (run_dir / "run.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "validation_predictions.json").write_text(json.dumps({"rows": best[4]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(run_dir)


if __name__ == "__main__":
    main()
