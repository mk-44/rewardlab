from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Any, Sequence, Optional
from rlhf.core.contracts import DataError, ConfigError
from rlhf.dataset.schema import UNASSIGNED_UID, PreferenceGroup

RecordFormat = Literal["pairwise", "pairwise_implicit", "kway", "arena"]
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "pairwise": ("prompt", "chosen", "rejected"),
    "pairwise_implicit": ("chosen", "rejected"),
    "kway": ("prompt", "responses"),
    "arena": ("prompt", "response_a", "response_b", "winner"),
}
#arena case
_WINNER_A = {"a", "model_a", "response_a", "0", "left"}
_WINNER_B = {"b", "model_b", "response_b", "1", "right"}
_WINNER_TIE = {"tie", "draw", "both", "neither", ""}

@dataclass(frozen = True)
class FieldMap:
    prompt : str = "prompt"
    chosen : str = "chosen"
    rejected : str = "rejected"
    responses : str = "responses"
    ranking : str = "ranking"
    scores : str = "scores"
    response_a : str = "response_a"
    response_b : str = "response_b"
    winner : str = "winner"

    def column(self, canonical : str) -> str:
        try:
            return getattr(self, canonical)
        except AttributeError:
            raise ConfigError(f"Unknown canonical field {canonical!r}. Known: {sorted(self.__dataclass_fields__)}")
    
    @classmethod
    def from_dict(cls, mapping : Optional[dict[str, str]]) -> FieldMap:
        if not mapping:
            return cls()
        valid = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - valid)
        if unknown:
            raise ConfigError(f"Unknown key(s) in `fields`: {unknown}. Valid: {sorted(valid)}")
        return cls(**mapping)

def _as_str(v: Any) -> str:
    return "" if v is None else str(v)

def _pick(record: dict, fields: FieldMap, canonical: str) -> Any:
    return record.get(fields.column(canonical))

def extract_slices(record: dict, slice_keys: Sequence[str]) -> dict[str, str]:
    return {k : str(record[k]) for k in slice_keys if k in record and record[k] is not None}

def _as_num_list(v : Any, cast, name : str) -> Optional[list]:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)):
        raise DataError(f"{name} must be a list. Got {type(v).__name__}")
    try:
        return [cast(x) for x in v]
    except (TypeError, ValueError) as e:
        raise DataError(f"{name} must contain numbers. {e}")


def validate_columns(record: dict, format: RecordFormat, fields: FieldMap) -> None:
    if format not in REQUIRED_COLUMNS:
        raise ConfigError(f"Unknown format {format!r}. Valid: {sorted(REQUIRED_COLUMNS)}")

    present = sorted(record)
    missing = [(c, fields.column(c)) for c in REQUIRED_COLUMNS[format]if fields.column(c) not in record]

    if missing:
        lines = "\n".join(f"{canon:12s} -> looking for column {col!r}" for canon, col in missing)
        raise DataError(
            f"format {format!r} needs columns that are not in the data.\n"
            f"    missing:\n{lines}\n"
            f"    columns present : {present}\n"
            f"    Fix: change `format:` or map the names under `fields:`."
        )

    if format == "kway":
        has_rank = fields.column("ranking") in record
        has_scores = fields.column("scores") in record
        if not (has_rank or has_scores):
            raise DataError(
                f"format 'kway' needs either {fields.column('ranking')!r} or {fields.column('scores')!r}.\ncolumns present : {present}")


def record_to_group(
    record : dict,
    format : RecordFormat,
    fields : FieldMap = FieldMap(),
    slice_keys : Sequence[str] = (),
    uid : int = UNASSIGNED_UID,
    meta : Optional[dict[str, Any]] = None,
) -> PreferenceGroup:
    slices = extract_slices(record, slice_keys)
    meta = meta or {}

    if format in ("pairwise", "pairwise_implicit"):
        prompt = _pick(record, fields, "prompt") if format == "pairwise" else ""
        chosen = _pick(record, fields, "chosen")
        rejected = _pick(record, fields, "rejected")
        if chosen is None or rejected is None:
            raise DataError("chosen or rejected is missing")
        return PreferenceGroup.from_pairwise(
            prompt = _as_str(prompt),
            chosen = _as_str(chosen),
            rejected = _as_str(rejected),
            slices = slices,
            meta = meta,
            uid = uid
        )

    if format == "arena":
        a = _pick(record, fields, "response_a")
        b = _pick(record, fields, "response_b")
        if a is None or b is None:
            raise DataError("response_a or response_b is missing")

        w = _as_str(_pick(record, fields, "winner")).strip().lower()
        if w in _WINNER_TIE:
            raise DataError("tie")
        if w in _WINNER_A:
            chosen, rejected = a, b
        elif w in _WINNER_B:
            chosen, rejected = b, a
        else:
            raise DataError(f"unknown winner value {w!r}. Known: {sorted(_WINNER_A | _WINNER_B | _WINNER_TIE)}")
        return PreferenceGroup.from_pairwise(
            prompt = _as_str(_pick(record, fields, "prompt")),
            chosen = _as_str(chosen),
            rejected = _as_str(rejected),
            slices = slices,
            meta = meta,
            uid = uid
        )

    if format == "kway":
        responses = _pick(record, fields, "responses")
        if not isinstance(responses, (list, tuple)):
            raise DataError(f"responses must be a list. Got {type(responses).__name__}")

        ranking = _pick(record, fields, "ranking")
        scores = _pick(record, fields, "scores")
        return PreferenceGroup(
            prompt = _as_str(_pick(record, fields, "prompt")),
            responses = [_as_str(r) for r in responses],
            ranking = _as_num_list(_pick(record, fields, "ranking"), int,   "ranking"),
            scores  = _as_num_list(_pick(record, fields, "scores"),  float, "scores"),
            slices = slices,
            meta = meta,
            uid = uid,
        )
    raise ConfigError(f"Unknown format {format!r}. Valid: {sorted(REQUIRED_COLUMNS)}")