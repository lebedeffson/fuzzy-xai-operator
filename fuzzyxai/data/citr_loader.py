from __future__ import annotations

from fuzzyxai.datasets import load_dataset_mode

from .dataset_record import DatasetRecord


def load_citr_dataset(
    *,
    mode: str = 'registry_mosmed_doctor_analysis',
    allow_fallback: bool = True,
) -> tuple[DatasetRecord, object]:
    """Load CIT registry dataset via dataset modes.

    If local file is missing and allow_fallback=True, falls back to built-in
    breast_cancer mode while preserving trace in metadata.
    """
    try:
        record, df = load_dataset_mode(mode)
        return record, df
    except FileNotFoundError:
        if not allow_fallback:
            raise
        record, df = load_dataset_mode('breast_cancer')
        record = DatasetRecord(
            name=f'citr_fallback_{mode}',
            source='fallback:breast_cancer',
            target_column=record.target_column,
            task_type=record.task_type,
            description='Fallback dataset when CIT registry local file is missing.',
            metadata={
                **record.metadata,
                'fallback': True,
                'fallback_reason': f'missing local file for {mode}',
                'requested_mode': mode,
            },
        )
        return record, df

