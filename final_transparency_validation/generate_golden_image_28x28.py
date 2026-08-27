"""Generates golden_image_28x28/ -- P18 item 14: a real, full-resolution
(28x28) image-XAI case, so the technical 8x8 sklearn-digits case in
golden_cnn/ stays what it always honestly was (a unit/integration-scale
pipeline test), while this case is the actual demonstration.

Uses Fashion-MNIST -- already a real benchmark in this framework's own
research code (framework/fuzzyxai/fuzzyxai/q1_validation/real_benchmarks.py),
reused here read-only via its own IDX download/parsing helpers rather than
reimplemented. The binary target (footwear classes 5/7/9 vs. the rest) is
the SAME derivation already used there, not a new one invented for this
script.

Saves the original 28x28 image, the full (un-aggregated) Integrated
Gradients tensor, and a proper full-resolution overlay -- the same real
pipeline as golden_cnn/, just at real image scale instead of 8x8.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework" / "fuzzyxai"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters.optional_v2 import TorchAdapter
from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.evidence import find_attribution_regions
from fuzzyxai.q1_validation.real_benchmarks import _download, _read_idx_images, _read_idx_labels
from torch import nn

OUT = Path(__file__).resolve().parent / "golden_image_28x28"
OUT.mkdir(exist_ok=True)
CACHE = Path(__file__).resolve().parent / ".fashion_mnist_cache"

# Same binary derivation as q1_validation/real_benchmarks.py::_load_fashion_mnist:
# footwear classes (5=sandal, 7=sneaker, 9=ankle boot) vs. every other apparel class.
FASHION_CLASS_NAMES_RU = {
    0: "футболка/топ", 1: "брюки", 2: "свитер", 3: "платье", 4: "пальто",
    5: "сандалия", 6: "рубашка", 7: "кроссовок", 8: "сумка", 9: "ботильон",
}
DOMAIN_LANGUAGE = {
    "classes": {"0": {"label": "не обувь"}, "1": {"label": "обувь (сандалия/кроссовок/ботильон)"}},
    "features": {},
    "actions": {},
}


class FashionCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(16 * 7 * 7, 2)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def _load_fashion_mnist_subset(train_n: int = 4000, test_n: int = 800) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
    names = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
    paths = [_download(f"{base}/{name}", CACHE / name) for name in names]
    train_images_full = _read_idx_images(paths[0])[:train_n]
    train_labels_full = _read_idx_labels(paths[1])[:train_n]
    test_images_full = _read_idx_images(paths[2])[:test_n]
    test_labels_full = _read_idx_labels(paths[3])[:test_n]
    x_train = train_images_full.astype(np.float32) / 255.0
    x_test = test_images_full.astype(np.float32) / 255.0
    y_train = np.isin(train_labels_full, (5, 7, 9)).astype(np.int64)
    y_test = np.isin(test_labels_full, (5, 7, 9)).astype(np.int64)
    return x_train, y_train, x_test, y_test, train_labels_full, test_labels_full


def main() -> None:
    torch.manual_seed(0)
    x_train, y_train, x_test, y_test, _train_original, test_original = _load_fashion_mnist_subset()

    net = FashionCNN()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    x_train_tensor = torch.as_tensor(x_train).unsqueeze(1)
    y_train_tensor = torch.as_tensor(y_train)
    batch_size = 128
    n = len(x_train_tensor)
    rng = np.random.default_rng(0)
    for _epoch in range(8):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_fn(net(x_train_tensor[idx]), y_train_tensor[idx]).backward()
            optimizer.step()
    net.eval()

    def input_transform(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(-1, 1, 28, 28))

    plan = ExplainPlan(domain_language=DOMAIN_LANGUAGE)

    with torch.no_grad():
        proba = torch.softmax(net(torch.as_tensor(x_test).unsqueeze(1)), dim=-1).numpy()
    predicted_labels = np.argmax(proba, axis=-1)
    correct_confident = np.where(predicted_labels == y_test)[0]
    if len(correct_confident) == 0:
        raise RuntimeError("no confidently-correct test object found -- cannot build the golden 28x28 case")
    object_index = int(correct_confident[np.argmax(proba[correct_confident].max(axis=1))])
    sample_image = x_test[object_index]
    flat = sample_image.flatten().tolist()

    convergence = []
    probe_result = None
    fx = None
    for n_steps in (16, 32, 64, 128, 256, 512):
        adapter = TorchAdapter(net, task="classification", input_transform=input_transform, ig_steps=n_steps)
        fx = FuzzyXAI.wrap(net, adapter=adapter, explain_plan=plan)
        probe_result = fx.explain_one(flat, raw_object=sample_image, feature_names=[f"px_{i}" for i in range(28 * 28)], object_id="fashion_p0")
        probe_maps = probe_result.view_model.layers.get("attribution_maps", [])
        if not probe_maps:
            raise RuntimeError(f"no attribution map was produced for n_steps={n_steps}")
        completeness = dict(probe_maps[0]["completeness"])
        convergence.append({
            "n_steps": n_steps,
            "F_target_x": completeness["F_target_x"],
            "F_target_baseline": completeness["F_target_baseline"],
            "input_output_delta": completeness["input_output_delta"],
            "attribution_sum": completeness["attribution_sum"],
            "absolute_residual": completeness["completeness_residual"],
            "relative_residual": completeness["completeness_relative_error"],
            "output_space": completeness["output_space"],
        })
    if probe_result is None or fx is None:  # pragma: no cover - fixed non-empty sweep
        raise RuntimeError("IG convergence sweep produced no result")
    (OUT / "ig_convergence.json").write_text(json.dumps(convergence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    attribution_maps = probe_result.view_model.layers.get("attribution_maps", [])
    if not attribution_maps:
        raise RuntimeError("no attribution map was produced -- cannot build the golden 28x28 case")
    attribution_array = attribution_maps[0]["attribution_array"]

    from fuzzyxai.evidence.attribution_map import _aggregate_channels

    regions = find_attribution_regions(attribution_array, image_shape=(28, 28), magnitude_percentile=80.0, min_pixels=6)
    heatmap, _ = _aggregate_channels(np.asarray(attribution_array), image_channels=1)
    region_sums = {name: float(heatmap[mask].sum()) for name, mask in regions.items()}

    result = fx.explain_one(
        flat,
        raw_object=sample_image,
        feature_names=[f"px_{i}" for i in range(28 * 28)],
        object_id="fashion_p0",
        region_masks=regions,
        evidence={"contributions": region_sums},
        dataset_version="fashion_mnist_footwear_vs_other_v1",
    )

    (OUT / "full_report_ru.txt").write_text(result.full_report(), encoding="utf-8")
    (OUT / "full_report_reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8")
    (OUT / "full_report_audit_ru.txt").write_text(result.full_report(level="audit"), encoding="utf-8")
    result.export_json(OUT / "result.json", detail="audit")
    (OUT / "audit.json").write_text(json.dumps(result.audit(), ensure_ascii=False, indent=2), encoding="utf-8")
    attribution_maps = result.view_model.layers.get("attribution_maps", [])
    attribution_evidence = json.loads(json.dumps(attribution_maps, default=str))
    (OUT / "attribution_evidence.json").write_text(json.dumps(attribution_evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "provenance.json").write_text(json.dumps(result.view_model.explanation_graph, ensure_ascii=False, indent=2), encoding="utf-8")

    if attribution_maps:
        overlay_b64 = attribution_maps[0]["attribution_png_base64"]
        (OUT / "attribution_overlay.png").write_bytes(base64.b64decode(overlay_b64))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(sample_image, cmap="gray")
    ax.axis("off")
    fig.savefig(OUT / "original.png", bbox_inches="tight")
    plt.close(fig)

    try:
        result.visualize(view="provenance", output=str(OUT / "provenance_action.png"))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - exporter records renderer failures
        (OUT / "visualization_error.txt").write_text(str(exc), encoding="utf-8")

    predicted_is_footwear = bool(int(result.prediction.predictions[0]))
    true_class_name = FASHION_CLASS_NAMES_RU[int(test_original[object_index])]
    final_completeness = attribution_maps[0]["completeness"] if attribution_maps else {}
    limitations_lines = [
        f"explanation_level = {result.explanation_level}",
        f"missing_channels = {list(result.missing_channels)}",
        f"action = {result.action}",
        f"Модель распознала: {'обувь' if predicted_is_footwear else 'не обувь'}; истинный класс Fashion-MNIST: {true_class_name}.",
        "dataset_version = fashion_mnist_footwear_vs_other_v1",
        "",
        "Метод: интегрированные градиенты (integrated_gradients), полный тензор атрибуции",
        "(та же форма, что и вход, 28x28 попиксельно) сохранён целиком в attribution_evidence.json.",
        f"Найдено алгоритмических регионов (connected components, порог 80-й перцентиль): {len(regions)}.",
        "Регионы НЕ являются фиксированной сеткой — их источник: пороговая обработка карты",
        "атрибуции + связные компоненты (4-связность), это раскрыто явно.",
        f"IG completeness: n_steps={final_completeness.get('n_steps')}, logit-space residual={final_completeness.get('completeness_residual')}, relative={final_completeness.get('completeness_relative_error')}.",
        "Convergence sweep 16/32/64/128/256/512 сохранён в ig_convergence.json.",
        "",
        "Это полноразмерный (28x28) кейс — golden_cnn/ (8x8, sklearn digits) остаётся",
        "техническим unit/integration-тестом конвейера, а не демонстрацией; это -- реальная",
        "демонстрация на общепринятом бенчмарке (Fashion-MNIST, уже используемом в",
        "q1_validation/real_benchmarks.py этого фреймворка).",
        "Обучение — на подвыборке (4000 train / 800 test) ради скорости прохода; это учебный,",
        "а не production-калиброванный классификатор.",
    ]
    (OUT / "limitations.txt").write_text("\n".join(limitations_lines) + "\n", encoding="utf-8")

    print("explanation_level:", result.explanation_level)
    print("missing_channels:", result.missing_channels)
    print("action:", result.action)
    print("regions found:", len(regions))
    print("predicted_is_footwear:", predicted_is_footwear, "true_class:", true_class_name)
    print("attribution shape:", attribution_maps[0]["shape"] if attribution_maps else None)


if __name__ == "__main__":
    main()
