from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, TextIO, Union

from rlhf.reward_model.core.contracts import ConfigError


def _sanitize(value: Any):
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    raise ConfigError(
        f"metric value {value!r} ({type(value).__name__}) is not a plain "
        f"int/float/str/bool/None — convert it before logging (tensors: .item())"
    )


class RunLogger:
    def __init__(self, out_dir: Union[str, Path], run_name: str,
                 is_main: bool = True, mode: str = "new",
                 console: Optional[TextIO] = None,
                 extra_sinks: Sequence[Callable[[dict], None]] = ()):
        if mode not in ("new", "resume"):
            raise ConfigError(f"mode must be 'new' or 'resume', got {mode!r}")
        self.is_main = is_main
        self.dir = Path(out_dir) / run_name
        self.n_rows = 0
        self.n_nonfinite = 0
        self._t0 = time.monotonic()
        self._console = console if console is not None else sys.stdout
        self._sinks = list(extra_sinks)
        self._metrics_f: Optional[TextIO] = None
        self._console_f: Optional[TextIO] = None
        self._warned_nonfinite = False

        if not self.is_main:
            return

        metrics_path = self.dir / "metrics.jsonl"
        if mode == "new" and metrics_path.exists():
            raise ConfigError(
                f"{metrics_path} already exists. Refusing to touch an existing "
                f"run. Pick a new run_name or pass mode='resume'."
            )
        self.dir.mkdir(parents=True, exist_ok=True)
        self._metrics_f = open(metrics_path, "a", encoding="utf-8")
        self._console_f = open(self.dir / "console.log", "a", encoding="utf-8")

    def stamp(self, name: str, payload: Mapping) -> None:
        if not self.is_main:
            return
        (self.dir / f"{name}.json").write_text(
            json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str) + "\n"
        )

    def log(self, step: Optional[int] = None, echo: bool = False, **metrics) -> None:
        row = {"step": step, "t": round(time.monotonic() - self._t0, 3)}
        for k, v in metrics.items():
            sv = _sanitize(v)
            if isinstance(sv, str) and sv in ("nan", "inf", "-inf"):
                self.n_nonfinite += 1
                if not self._warned_nonfinite:
                    self.say(f"WARNING non-finite metric {k}={sv} at step {step} "
                             f"(stored as string; counting silently from now on)")
                    self._warned_nonfinite = True
            row[k] = sv
        self.n_rows += 1
        for sink in self._sinks:
            sink(row)
        if self._metrics_f is not None:
            self._metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._metrics_f.flush()
        if echo:
            body = "   ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                              for k, v in metrics.items())
            self.say(f"step {step:>8}   {body}" if step is not None else body)

    def say(self, msg: str) -> None:
        if not self.is_main:
            return
        print(msg, file=self._console)
        if self._console_f is not None:
            self._console_f.write(msg + "\n")
            self._console_f.flush()

    class _Timer:
        def __init__(self, logger: "RunLogger", name: str):
            self.logger, self.name = logger, name

        def __enter__(self):
            self.t0 = time.monotonic()
            return self

        def __exit__(self, *exc):
            self.logger.log(step=None, **{f"time/{self.name}": time.monotonic() - self.t0})
            return False

    def timer(self, name: str) -> "RunLogger._Timer":
        return RunLogger._Timer(self, name)

    def close(self) -> None:
        for f in (self._metrics_f, self._console_f):
            if f is not None:
                f.close()
        self._metrics_f = self._console_f = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def read_metrics(path: Union[str, Path]) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out
