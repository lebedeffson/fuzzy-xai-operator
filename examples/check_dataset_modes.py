from __future__ import annotations

from fuzzyxai.datasets import list_dataset_modes, load_dataset_mode
from fuzzyxai.datasets.registry_mosmed_doctor_analysis import DEFAULT_CANDIDATE_PATHS as MOSMED_PATHS
from fuzzyxai.datasets.registry_programs import DEFAULT_CANDIDATE_PATHS as PROGRAMS_PATHS
from fuzzyxai.datasets.registry_steel_ir import DEFAULT_CANDIDATE_PATHS as STEEL_PATHS

VALIDATES = {
    'breast_cancer': 'risk observer baseline',
    'diabetes_binary': 'borderline uncertainty',
    'wine_risk': 'transferability',
    'synthetic_ruptures': 'controlled ruptures',
    'registry_programs': 'registry records',
    'registry_mosmed_doctor_analysis': 'doctor/model audit',
    'registry_steel_ir': 'industrial transferability',
}

DOMAIN = {
    'breast_cancer': 'medical',
    'diabetes_binary': 'medical',
    'wine_risk': 'tabular',
    'synthetic_ruptures': 'diagnostic',
    'registry_programs': 'text-tabular',
    'registry_mosmed_doctor_analysis': 'medical audit',
    'registry_steel_ir': 'industrial CV',
}

EXPECTED_PATHS = {
    'registry_programs': str(PROGRAMS_PATHS[0]),
    'registry_mosmed_doctor_analysis': str(MOSMED_PATHS[0]),
    'registry_steel_ir': str(STEEL_PATHS[0]),
}


def main() -> None:
    print('dataset_mode\tstatus\trows\tdomain\tvalidates\tdetails')
    for spec in list_dataset_modes():
        try:
            record, df = load_dataset_mode(spec.key)
            print(f"{spec.key}\tREADY\t{len(df)}\t{DOMAIN.get(spec.key, spec.domain)}\t{VALIDATES.get(spec.key, spec.purpose)}\t{record.source}")
        except FileNotFoundError as exc:
            expected = EXPECTED_PATHS.get(spec.key, str(exc))
            print(f"{spec.key}\tMISSING\t-\t{DOMAIN.get(spec.key, spec.domain)}\t{VALIDATES.get(spec.key, spec.purpose)}\t{expected}")
        except Exception as exc:
            print(f"{spec.key}\tERROR\t-\t{DOMAIN.get(spec.key, spec.domain)}\t{VALIDATES.get(spec.key, spec.purpose)}\t{type(exc).__name__}: {exc}")


if __name__ == '__main__':
    main()
