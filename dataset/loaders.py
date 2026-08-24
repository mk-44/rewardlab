from __future__ import annotations
import csv
import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union
from rlhf.core.contracts import ConfigError, DataError
from rlhf.dataset.formats import FieldMap, RecordFormat, record_to_group, validate_columns
from rlhf.dataset.schema import PreferenceGroup

Source = Union[str, Path, Sequence[dict], Any]

def sha256_of(path : Union[str, Path], chunk : int = 1 << 20) -> str:
    "File ka fingerprint. [O(file size) time aur O(chunk) memory]"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda : f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def iter_records(source: Source, json_key: Optional[str] = None) -> Iterator[dict]:
    "Kisi bhi source se raw dicts stream karo. json jsonl csv parquet list ya HF dataset. [O(1) memory jahan ho sake]"
    if isinstance(source, (list, tuple)):
        yield from source
        return

    if hasattr(source, "column_names"):
        for row in source:
            yield dict(row)
        return

    path = Path(source)
    if not path.exists():
        raise DataError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".jsonl":
        with open(path, "r", encoding = "utf-8") as f:
            for l_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise DataError(f"Bad JSON at {path}:{l_no}: {e}")
        return

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if json_key is None:
                raise ConfigError(f"{path} holds a dict, not a list. Set `json_key` to the key that holds the records.\nkeys present : {sorted(data)[:20]}")
            if json_key not in data:
                raise ConfigError(f"json_key={json_key!r} not found in {path}.\nkeys present : {sorted(data)[:20]}")
            data = data[json_key]

        if not isinstance(data, list):
            raise DataError(f"{path} must hold a list of records. Got {type(data).__name__}.")
        yield from data
        return

    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return

    if ext == ".parquet":
        try:
            import pandas as pd
        except ImportError:
            raise ConfigError("parquet needs pandas. pip install pandas")
        yield from pd.read_parquet(path).to_dict("records")
        return

    raise ConfigError(f"Unsupported file type {ext!r}. Supported: .json .jsonl .csv .parquet")


@dataclass
class LoadReport:
    "Load ka pura hisaab. Seedha run.json aur audit report mein jaata hai."
    source : str
    format : str
    sha256 : Optional[str] = None
    slice_keys : list[str] = field(default_factory=list)

    n_records : int = 0
    n_groups : int = 0
    n_dropped : int = 0
    drop_reasons : Counter = field(default_factory=Counter)
    examples : list[str] = field(default_factory=list)
    warnings : list[str] = field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        "Kitna hissa gira. Zero records pe zero."
        return self.n_dropped / self.n_records if self.n_records else 0.0

    def summary(self) -> str:
        "Console ke liye ek chhota block."
        lines = [
            f"  source      : {self.source}",
            f"  sha256      : {(self.sha256 or 'n/a')[:16]}",
            f"  format      : {self.format}",
            f"  records     : {self.n_records:,}",
            f"  groups      : {self.n_groups:,}",
            f"  dropped     : {self.n_dropped:,}  ({self.drop_rate:.2%})",
            f"  slice_keys  : {self.slice_keys or 'none'}",
        ]
        
        for reason, n in self.drop_reasons.most_common():
            lines.append(f"      {n:>7,}  {reason}")
        
        for w in self.warnings:
            lines.append(f"  WARNING     : {w}")
        
        return "\n".join(lines)


@dataclass
class LoadResult:
    "Groups aur unka hisaab saath mein. Taaki report kho na jaaye."
    groups : list[PreferenceGroup]
    report : LoadReport


def load_groups(
    source : Source,
    format : RecordFormat,
    fields : FieldMap = FieldMap(),
    slice_keys : Sequence[str] = (),
    json_key : Optional[str] = None,
    max_drop_rate : float = 0.05,
    start_uid : int = 0,
    max_examples : int = 5,
) -> LoadResult:
    "Kuch bhi padho aur groups banao. Format declared hai toh yahan koi guess nahi. [O(N + T)]"
    if not 0.0 <= max_drop_rate <= 1.0:
        raise ConfigError(f"max_drop_rate must be in [0, 1]. Got {max_drop_rate}.")

    is_path = isinstance(source, (str, Path)) and Path(source).exists()
    report = LoadReport(
        source = str(source) if is_path else f"<{type(source).__name__}>",
        format = format,
        sha256 = sha256_of(source) if is_path else None,
        slice_keys = list(slice_keys),
    )
    stream = iter_records(source, json_key = json_key)

    try:
        first = next(stream)
    except StopIteration:
        raise DataError(f"No records found in {report.source}.")
    validate_columns(first, format, fields)

    stream = itertools.chain([first], stream)
    groups: list[PreferenceGroup] = []
    uid = start_uid

    for row_no, record in enumerate(stream):
        report.n_records += 1
        try:
            groups.append(
                record_to_group(
                    record,
                    format = format,
                    fields = fields,
                    slice_keys = slice_keys,
                    uid = uid,
                    meta = {"row": row_no},
                )
            )
            uid += 1
            report.n_groups += 1
        except DataError as e:
            report.n_dropped += 1
            reason = str(e).splitlines()[0][:80]
            report.drop_reasons[reason] += 1
            if len(report.examples) < max_examples:
                report.examples.append(f"row {row_no}: {reason}")

    if report.drop_rate > max_drop_rate:
        raise DataError(
            f"Dropped {report.n_dropped:,} of {report.n_records:,} records "
            f"({report.drop_rate:.1%}), above max_drop_rate={max_drop_rate:.1%}.\n"
            f"This is not noise. Check `format` and `fields`.\n"
            f"    format   : {format}\n"
            f"    reasons  : {dict(report.drop_reasons.most_common(5))}\n"
            f"    examples : {report.examples}"
        )

    if report.n_groups == 0:
        raise DataError(f"No usable groups built from {report.source}.")

    return LoadResult(groups = groups, report = report)