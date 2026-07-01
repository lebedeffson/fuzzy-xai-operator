from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .dataset_loader import infer_file_format, load_table_dataset
from .dataset_record import DatasetRecord


class CITRegistryDatasetClient:
    """Loader for CIT Registry datasets.

    The registry page is dynamic, so v1 supports direct file URLs and local files
    downloaded manually from https://registry.cit.gov.ru/datasets.
    """

    def __init__(self, cache_dir: str | Path = 'data/cit_registry') -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def from_local_file(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        target_column: str | None = None,
        description: str = '',
    ) -> DatasetRecord:
        path = Path(path)
        fmt = infer_file_format(path)
        return DatasetRecord(
            name=name or path.stem,
            source='registry.cit.gov.ru',
            local_path=path,
            file_format=fmt,
            target_column=target_column,
            description=description,
            metadata={'load_mode': 'local_file'},
        )

    @staticmethod
    def normalize_direct_url(url: str) -> str:
        """Accept direct file URLs and common GitHub blob links."""
        parsed = urlparse(url)
        if parsed.netloc == 'github.com' and '/blob/' in parsed.path:
            owner_repo, _, file_path = parsed.path.lstrip('/').partition('/blob/')
            return f'https://raw.githubusercontent.com/{owner_repo}/{file_path}'
        return url

    def download_file(self, url: str, *, name: str | None = None, target_column: str | None = None) -> DatasetRecord:
        normalized_url = self.normalize_direct_url(url)
        parsed = urlparse(normalized_url)
        filename = Path(parsed.path).name or 'dataset.csv'
        local_path = self.cache_dir / filename
        request = Request(normalized_url, headers={'User-Agent': 'fuzzyxai-dataset-observer/1.0'})
        with urlopen(request, timeout=60) as response:
            local_path.write_bytes(response.read())
        fmt = infer_file_format(local_path)
        return DatasetRecord(
            name=name or local_path.stem,
            source=parsed.netloc or 'direct-url',
            local_path=local_path,
            url=normalized_url,
            file_format=fmt,
            target_column=target_column,
            metadata={'load_mode': 'direct_url', 'original_url': url},
        )

    def load_dataframe(self, record: DatasetRecord):
        if record.local_path is None:
            raise ValueError('DatasetRecord has no local_path')
        return load_table_dataset(record.local_path)
