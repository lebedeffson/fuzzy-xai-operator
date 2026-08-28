from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from chapter6_medical_validation.ecg_ptbxl.src.model import build_ecg_resnet1d
from chapter6_medical_validation.shared.calibration import fit_temperature, softmax
from chapter6_medical_validation.shared.hashing import sha256_file
from chapter6_medical_validation.shared.metrics_common import binary_metrics

ROOT = Path(__file__).resolve().parents[1]


class PreparedDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.signals, self.labels, self.indices = signals, labels, indices
        self.mean, self.std = mean[:, None], std[:, None]
    def __len__(self) -> int:
        return len(self.indices)
    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item]); value = (np.asarray(self.signals[index]) - self.mean) / self.std
        return torch.tensor(value, dtype=torch.float32), torch.tensor(int(self.labels[index])), index


def evaluate(model, loader, device):
    model.eval(); losses, logits_all, labels_all, indices_all = [], [], [], []
    with torch.no_grad():
        for inputs, labels, indices in loader:
            logits = model(inputs.to(device)); losses.append(float(torch.nn.functional.cross_entropy(logits, labels.to(device)).item()))
            logits_all.extend(logits.cpu().tolist()); labels_all.extend(labels.tolist()); indices_all.extend(indices.tolist())
    return float(np.mean(losses)), np.asarray(logits_all), np.asarray(labels_all), np.asarray(indices_all)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); args = parser.parse_args()
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root: raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    prepared = Path(data_root) / "ecg" / "ptb-xl-1.0.3" / "prepared"
    config = yaml.safe_load((ROOT / "configs" / "model_ecg_resnet1d.yaml").read_text())
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    signals = np.load(prepared / "signals.npy", mmap_mode="r"); labels = np.load(prepared / "labels.npy"); folds = np.load(prepared / "folds.npy"); ecg_ids = np.load(prepared / "ecg_ids.npy")
    stats = json.loads((prepared / "normalization.json").read_text()); mean, std = np.asarray(stats["lead_mean"]), np.asarray(stats["lead_std"])
    indices = {"train": np.flatnonzero(folds <= 8), "validation": np.flatnonzero(folds == 9), "test": np.flatnonzero(folds == 10)}
    datasets = {name: PreparedDataset(signals, labels, value, mean, std) for name, value in indices.items()}
    loaders = {name: DataLoader(value, batch_size=int(config["batch_size"]), shuffle=name == "train", num_workers=0, pin_memory=True) for name, value in datasets.items()}
    device = torch.device("cuda"); model = build_ecg_resnet1d(tuple(config["channels"]), int(config["blocks_per_stage"])).to(device)
    counts = np.bincount(labels[indices["train"]], minlength=2); weights = len(indices["train"]) / (2 * counts)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device)); optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optimizer"]["learning_rate"]), weight_decay=float(config["optimizer"]["weight_decay"]))
    run_id = f"ecg-resnet1d-seed-{args.seed}"; run_dir = ROOT / "outputs" / "runs" / run_id; run_dir.mkdir(parents=True, exist_ok=False)
    best_loss, no_improvement, history, probe_history = float("inf"), 0, [], []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train(); train_loss = []
        for inputs, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True); output = model(inputs.to(device)); loss = criterion(output, target.to(device)); loss.backward(); optimizer.step(); train_loss.append(float(loss.item()))
        val_loss, val_logits, val_labels, val_indices = evaluate(model, loaders["validation"], device)
        val_prob = softmax(val_logits)[:, 1]; val_metrics = binary_metrics(val_labels, val_prob)
        probe_probability = softmax(val_logits[:1])[0]
        probe_prediction = int(np.argmax(probe_probability)); probe_truth = int(val_labels[0])
        probe_history.append({"epoch": epoch, "predicted_class": probe_prediction, "confidence": float(probe_probability[probe_prediction]), "correct": probe_prediction == probe_truth, "loss": float(-np.log(max(float(probe_probability[probe_truth]), 1e-15))), "loss_status": "measured_object_negative_log_likelihood"})
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_loss)), "validation_loss": val_loss, "validation_auroc": val_metrics["auroc"]})
        print(history[-1], flush=True)
        if val_loss < best_loss - 1e-8:
            best_loss, no_improvement = val_loss, 0; torch.save({"state_dict": model.state_dict(), "seed": args.seed, "epoch": epoch, "run_id": run_id}, run_dir / "best.pt")
        else:
            no_improvement += 1
            if no_improvement >= int(config["patience"]): break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["state_dict"])
    _, val_logits, val_labels, val_indices = evaluate(model, loaders["validation"], device); calibration = fit_temperature(val_logits, val_labels); temperature = float(calibration["temperature"])
    _, test_logits, test_labels, test_indices = evaluate(model, loaders["test"], device)
    probe_index = int(val_indices[0]); probe_object_id = f"ptbxl-{int(ecg_ids[probe_index])}"
    artifacts = {"run_id": run_id, "seed": args.seed, "best_epoch": checkpoint["epoch"], "history": history, "training_probe": {"object_id": probe_object_id, "prepared_index": probe_index, "ecg_id": int(ecg_ids[probe_index]), "truth": int(val_labels[0]), "history_through_final_checkpoint": probe_history[: int(checkpoint["epoch"])], "epoch_source": "measured on fixed validation object after each completed epoch", "final_checkpoint_ref": f"best.pt:epoch:{int(checkpoint['epoch'])}"}, "checkpoint_sha256": sha256_file(run_dir / "best.pt"), "calibration": calibration, "validation_metrics_uncalibrated": binary_metrics(val_labels, softmax(val_logits)[:, 1]), "validation_metrics_calibrated": binary_metrics(val_labels, softmax(val_logits, temperature)[:, 1]), "test_metrics_uncalibrated": binary_metrics(test_labels, softmax(test_logits)[:, 1]), "test_metrics_calibrated": binary_metrics(test_labels, softmax(test_logits, temperature)[:, 1])}
    (run_dir / "run.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    np.savez_compressed(run_dir / "validation_predictions.npz", logits=val_logits, labels=val_labels, prepared_indices=val_indices); np.savez_compressed(run_dir / "test_predictions.npz", logits=test_logits, labels=test_labels, prepared_indices=test_indices)
    print(run_dir)


if __name__ == "__main__": main()
