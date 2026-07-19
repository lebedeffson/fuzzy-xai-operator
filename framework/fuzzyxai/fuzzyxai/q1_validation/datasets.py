"""Real-benchmark registry and controlled-perturbation metadata.

Raw benchmark data are never bundled automatically. Acquisition jobs record
the observed hash and license boundary before a run can be marked measured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class Q1DatasetSpec:
    dataset_id: str
    modality: Literal["tabular", "image", "text", "timeseries"]
    task: str
    source: str
    license: str
    version: str
    minimum_objects: int
    acquisition: str
    redistribution: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


REAL_BENCHMARKS: tuple[Q1DatasetSpec, ...] = (
    Q1DatasetSpec(
        "uci_covertype",
        "tabular",
        "multiclass_classification",
        "https://archive.ics.uci.edu/dataset/31/covertype",
        "CC BY 4.0 (verify against downloaded dataset card)",
        "UCI-31",
        10_000,
        "sklearn.datasets.fetch_covtype",
        "reference_only",
        ("forest-cover benchmark; not a safety-domain validation",),
    ),
    Q1DatasetSpec(
        "fashion_mnist",
        "image",
        "multiclass_classification",
        "https://github.com/zalandoresearch/fashion-mnist",
        "MIT",
        "fashion-mnist-v1",
        10_000,
        "official gzip files",
        "reference_only",
        ("grayscale apparel images; no medical generalization",),
    ),
    Q1DatasetSpec(
        "20newsgroups",
        "text",
        "multiclass_classification",
        "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html",
        "dataset content has source-specific rights; loader is BSD-3-Clause",
        "sklearn-cache-versioned",
        10_000,
        "sklearn.datasets.fetch_20newsgroups",
        "do_not_redistribute_raw_text",
        ("historical posts may contain personal or offensive content", "license must be reviewed before redistribution"),
    ),
    Q1DatasetSpec(
        "ucr_electric_devices",
        "timeseries",
        "multiclass_classification",
        "https://www.timeseriesclassification.com/description.php?Dataset=ElectricDevices",
        "dataset-specific UCR terms; verify before redistribution",
        "UCR-ElectricDevices",
        10_000,
        "UCR archive download",
        "reference_only",
        ("household electricity benchmark; dataset-specific terms apply",),
    ),
)


def dataset_registry() -> tuple[dict[str, object], ...]:
    return tuple(item.to_dict() for item in REAL_BENCHMARKS)
