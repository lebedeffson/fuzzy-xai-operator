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

from chapter6_medical_validation.brain_allen.src.model import build_inception_binary
from chapter6_medical_validation.brain_allen.src.preprocessing import preprocess_patch
from chapter6_medical_validation.shared.calibration import fit_temperature, softmax
from chapter6_medical_validation.shared.hashing import sha256_file
from chapter6_medical_validation.shared.metrics_common import binary_metrics

ROOT = Path(__file__).resolve().parents[1]


class PatchDataset(Dataset):
    def __init__(self, patches, metadata, indexes, scale: float, split: str, seed: int):
        self.patches, self.metadata, self.indexes, self.scale, self.split, self.seed = patches, metadata, indexes, scale, split, seed
    def __len__(self): return len(self.indexes)
    def __getitem__(self, item):
        index = int(self.indexes[item]); image, _display, _trace = preprocess_patch(np.asarray(self.patches[index]), scale=self.scale, split=self.split, seed=self.seed, object_index=index)
        return image, torch.tensor(int(self.metadata[index]["label"])), index


def evaluate(model, loader, device):
    model.eval(); logits_all, labels_all, indices_all, losses = [], [], [], []
    with torch.no_grad():
        for inputs, labels, indices in loader:
            logits = model(inputs.to(device)); losses.append(float(torch.nn.functional.cross_entropy(logits, labels.to(device)).item())); logits_all.extend(logits.cpu().tolist()); labels_all.extend(labels.tolist()); indices_all.extend(indices.tolist())
    return float(np.mean(losses)), np.asarray(logits_all), np.asarray(labels_all), np.asarray(indices_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prepared-name", default="prepared")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--model-config", default="model_inceptionv3.yaml")
    parser.add_argument("--run-prefix", default="brain-inceptionv3")
    args = parser.parse_args()
    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root: raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set")
    prepared = Path(data_root) / "brain" / "allen_ccf_25um" / args.prepared_name; patches = np.load(prepared / "patches.npy", mmap_mode="r"); metadata = json.loads((prepared / "patches.json").read_text())
    config_path = ROOT / "configs" / args.model_config
    config = yaml.safe_load(config_path.read_text()); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    indices = {split: np.asarray([index for index, item in enumerate(metadata) if item["split"] == split]) for split in ("train", "validation", "test")}
    scale = float(np.percentile(np.asarray(patches[indices["train"]]), 99.5)); statistics = {"status": "fitted_train_only", "intensity_percentile": 99.5, "scale": scale}
    datasets = {split: PatchDataset(patches, metadata, value, scale, split, args.seed) for split, value in indices.items()}; loaders = {split: DataLoader(value, batch_size=int(config["batch_size"]), shuffle=split == "train", num_workers=0) for split, value in datasets.items()}
    device = torch.device("cuda"); model = build_inception_binary(pretrained=True).to(device); labels_train = np.asarray([metadata[int(index)]["label"] for index in indices["train"]]); counts = np.bincount(labels_train, minlength=2); criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(labels_train) / (2 * counts), dtype=torch.float32, device=device)); optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["optimizer"]["learning_rate"]), weight_decay=float(config["optimizer"]["weight_decay"]))
    run_id = f"{args.run_prefix}-seed-{args.seed}"; run_dir = ROOT / args.output_root / "runs" / run_id; run_dir.mkdir(parents=True, exist_ok=False); best_loss, stale, history = float("inf"), 0, []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train(); epoch_losses = []
        for inputs, labels, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True); output = model(inputs.to(device)); loss = criterion(output, labels.to(device)); loss.backward(); optimizer.step(); epoch_losses.append(float(loss.item()))
        val_loss, val_logits, val_labels, _ = evaluate(model, loaders["validation"], device); metrics = binary_metrics(val_labels, softmax(val_logits)[:, 1]); history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "validation_loss": val_loss, "validation_macro_f1": metrics["f1"]}); print(history[-1], flush=True)
        if val_loss < best_loss - 1e-8: best_loss, stale = val_loss, 0; torch.save({"state_dict": model.state_dict(), "seed": args.seed, "epoch": epoch, "run_id": run_id, "preprocessing": statistics}, run_dir / "best.pt")
        else:
            stale += 1
            if stale >= int(config["patience"]): break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False); model.load_state_dict(checkpoint["state_dict"]); _, val_logits, val_labels, val_idx = evaluate(model, loaders["validation"], device); calibration = fit_temperature(val_logits, val_labels); temperature = float(calibration["temperature"]); _, test_logits, test_labels, test_idx = evaluate(model, loaders["test"], device)
    result = {"run_id": run_id, "seed": args.seed, "protocol_id": config.get("protocol_id", "brain_v1_pilot"), "prepared_manifest_sha256": sha256_file(prepared / "dataset_manifest.json"), "model_config_sha256": sha256_file(config_path), "best_epoch": checkpoint["epoch"], "checkpoint_sha256": sha256_file(run_dir / "best.pt"), "preprocessing": statistics, "history": history, "calibration": calibration, "validation_metrics": binary_metrics(val_labels, softmax(val_logits, temperature)[:, 1]), "test_metrics": binary_metrics(test_labels, softmax(test_logits, temperature)[:, 1])}; (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n"); np.savez_compressed(run_dir / "validation_predictions.npz", logits=val_logits, labels=val_labels, prepared_indices=val_idx); np.savez_compressed(run_dir / "test_predictions.npz", logits=test_logits, labels=test_labels, prepared_indices=test_idx); print(run_dir)


if __name__ == "__main__": main()
