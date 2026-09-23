from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

PLACEHOLDERS = ("{prompt}", "{constraints}", "{response}")
DROPPABLE = ("temperature", "reasoning_effort", "response_format")
RETRY_STATUS = {429, 500, 502, 503, 504}
REASONING = re.compile(r"^(gpt-5|o\d)")


def read_jsonl(path : Path) -> List[dict]:
    with open(path, encoding = "utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path : Path, rows : Sequence[dict]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding = "utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii = False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path : Path, rows : Sequence[dict]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii = False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_env_file(path : Path, environ = os.environ) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in environ:
            environ[k] = v


def api_key(env_var : str, env_file : Path, environ = os.environ) -> str:
    load_env_file(env_file, environ)
    key = environ.get(env_var, "").strip()
    if not key:
        raise SystemExit(f"{env_var} is not set. put it in {env_file} or the environment")
    return key


def check_template(template : str) -> None:
    missing = [p for p in PLACEHOLDERS if p not in template]
    if missing:
        raise SystemExit(f"prompt file lacks placeholders {' '.join(missing)}")


def split_constraints(text : str, sep : str) -> List[str]:
    return [c.strip() for c in str(text).split(sep) if c.strip()]


def fill(template : str, prompt : str, constraints : Sequence[str], response : str) -> str:
    listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(constraints))
    out = template.replace("{n_constraints}", str(len(constraints)))
    return out.replace("{prompt}", prompt).replace("{constraints}", listed).replace("{response}", response)


def param_style(model : str, style : str) -> str:
    if style != "auto":
        return style
    return "reasoning" if REASONING.match(model) else "temperature"


def payload_for(model : str, text : str, style : str, reasoning_effort : str, temperature : float, json_mode : bool, dropped : Sequence[str]) -> dict:
    p = {"model" : model, "messages" : [{"role" : "user", "content" : text}]}
    if style == "reasoning":
        p["reasoning_effort"] = reasoning_effort
    elif style == "temperature":
        p["temperature"] = temperature
    if json_mode:
        p["response_format"] = {"type" : "json_object"}
    for name in dropped:
        p.pop(name, None)
    return p


def strip_fences(text : str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3]
    return t.strip()


def parse_reply(raw : str, constraints : Sequence[str]) -> dict:
    try:
        data = json.loads(strip_fences(raw))
    except ValueError as e:
        raise ValueError(f"reply is not json ({e})")
    if not isinstance(data, dict) or not isinstance(data.get("constraints"), list):
        raise ValueError("reply has no constraints list")
    items = data["constraints"]
    if len(items) != len(constraints):
        raise ValueError(f"reply has {len(items)} constraints but the row has {len(constraints)}")
    verdicts = []
    for c, it in zip(constraints, items):
        if not isinstance(it, dict) or not isinstance(it.get("pass"), bool):
            raise ValueError("a constraint entry has no boolean pass")
        v = {"constraint" : c, "pass" : it["pass"], "reason" : str(it.get("reason", ""))}
        if isinstance(it.get("requirement"), str):
            v["requirement"] = it["requirement"]
        verdicts.append(v)
    flags = {k : v for k, v in data.items() if k != "constraints" and isinstance(v, bool)}
    return {"constraints" : verdicts, "flags" : flags}


class Pacer:
    def __init__(self, min_interval : float):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.min_interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


def call(post : Callable, url : str, headers : dict, payload : dict, timeout : float, retries : int, dropped : List[str], pace : Callable = lambda : None):
    attempt = 0
    while True:
        status, body, err = None, None, None
        pace()
        try:
            r = post(url, headers = headers, json = payload, timeout = timeout)
            status = r.status_code
            try:
                body = r.json()
            except ValueError:
                body = None
        except Exception as e:
            err = str(e)
        if status == 200 and isinstance(body, dict):
            content = body["choices"][0]["message"]["content"]
            return content, body.get("usage") or {}
        if status == 400 and body is not None:
            msg = json.dumps(body)
            hit = next((n for n in DROPPABLE if n in payload and n in msg), None)
            if hit is None:
                raise RuntimeError(f"http 400 {msg[:300]}")
            payload = {k : v for k, v in payload.items() if k != hit}
            dropped.append(hit)
            continue
        if status is None or status in RETRY_STATUS:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"http {status} after {retries} retries {err or ''}".rstrip())
            time.sleep(min(60.0, 2.0 ** attempt) + random.random())
            continue
        raise RuntimeError(f"http {status} {json.dumps(body)[:300] if body is not None else err}")


@dataclass
class JudgeReport:
    out : str = ""
    model : str = ""
    target : str = ""
    rows : int = 0
    skipped : int = 0
    judged : int = 0
    errors : int = 0
    all_pass : int = 0
    n_pass_sum : int = 0
    n_constraints_sum : int = 0
    flags : Dict[str, int] = field(default_factory = dict)
    input_tokens : int = 0
    output_tokens : int = 0
    seconds : float = 0.0
    aborted : bool = False
    consecutive_http_failures : int = 0


def load_existing(out : Path, settings : dict):
    if not out.exists():
        return {}, 0
    good, dropped = {}, 0
    for r in read_jsonl(out):
        for k, v in settings.items():
            if r.get(k) != v:
                raise SystemExit(f"{out} holds rows with {k} = {r.get(k)!r} but this run has {v!r}. use another out path")
        if r.get("judge_error") is None and r.get("judge_all_pass") is not None:
            good[int(r["idx"])] = r
        else:
            dropped += 1
    return good, dropped


def judge_row(row : dict, idx : int, ctx : dict, dropped : List[str]) -> dict:
    a = ctx["args"]
    constraints = split_constraints(row[a.constraints_field], a.sep)
    base = {
        "judge_constraints" : None, "judge_n_pass" : None, "judge_all_pass" : None,
        "judge_model" : a.model, "judge_prompt_sha256" : ctx["sha"], "judge_target" : a.target,
        "judge_raw" : None, "judge_error" : None, "judge_input_tokens" : None, "judge_output_tokens" : None,
    }
    out = {**row, "idx" : idx, **base}
    if not constraints:
        out["judge_error"] = "row has no constraints"
        return out
    text = fill(ctx["template"], str(row[a.prompt_field]), constraints, str(row[a.target]))
    payload = payload_for(a.model, text, ctx["style"], a.reasoning_effort, a.temperature, not a.no_json_mode, dropped)
    try:
        raw, usage = call(ctx["post"], ctx["url"], ctx["headers"], payload, a.timeout, a.retries, dropped, ctx["pace"])
    except RuntimeError as e:
        out["judge_error"] = str(e)
        return out
    out["judge_raw"] = raw
    out["judge_input_tokens"] = usage.get("prompt_tokens")
    out["judge_output_tokens"] = usage.get("completion_tokens")
    try:
        parsed = parse_reply(raw, constraints)
    except ValueError as e:
        out["judge_error"] = str(e)
        return out
    n_pass = sum(1 for v in parsed["constraints"] if v["pass"])
    out["judge_constraints"] = parsed["constraints"]
    out["judge_n_pass"] = n_pass
    out["judge_all_pass"] = n_pass == len(constraints)
    for k, v in parsed["flags"].items():
        out[f"judge_{k}"] = v
    return out


def tally(rep : JudgeReport, r : dict) -> None:
    rep.judged += 1
    if r["judge_error"] is not None:
        rep.errors += 1
        rep.consecutive_http_failures = rep.consecutive_http_failures + 1 if r["judge_error"].startswith("http") else 0
        return
    rep.consecutive_http_failures = 0
    rep.all_pass += int(r["judge_all_pass"])
    rep.n_pass_sum += r["judge_n_pass"]
    rep.n_constraints_sum += len(r["judge_constraints"])
    rep.input_tokens += r["judge_input_tokens"] or 0
    rep.output_tokens += r["judge_output_tokens"] or 0
    for k, v in r.items():
        if k.startswith("judge_") and isinstance(v, bool) and k != "judge_all_pass":
            rep.flags[k] = rep.flags.get(k, 0) + int(v)


def run(args, post : Optional[Callable] = None, environ = os.environ) -> JudgeReport:
    if args.workers < 1:
        raise SystemExit("workers must be >= 1")
    if args.retries < 0:
        raise SystemExit("retries must be >= 0")
    if args.min_interval < 0:
        raise SystemExit("min interval must be >= 0")
    if args.max_consecutive_failures < 1:
        raise SystemExit("max consecutive failures must be >= 1")
    template = Path(args.prompt_file).read_text(encoding = "utf-8")
    check_template(template)
    sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
    out = Path(args.out)
    settings = {"judge_model" : args.model, "judge_prompt_sha256" : sha, "judge_target" : args.target}

    rows = read_jsonl(Path(args.rows))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"{args.rows} has no rows")
    for name in (args.prompt_field, args.constraints_field, args.target):
        if name not in rows[0]:
            raise SystemExit(f"rows lack field {name!r}")

    good, dropped_rows = load_existing(out, settings)
    if dropped_rows:
        write_jsonl(out, [good[i] for i in sorted(good)])
        print(f"{out}: kept {len(good)} judged rows and dropped {dropped_rows} rows with errors for a retry", flush = True)
    todo = [(i, r) for i, r in enumerate(rows) if i not in good]
    rep = JudgeReport(out = str(out), model = args.model, target = args.target, rows = len(rows), skipped = len(good))
    if not todo:
        print(f"{out}: all {len(rows)} rows judged. nothing to do", flush = True)
        return rep

    key = api_key(args.api_key_env, Path(args.env_file), environ)
    if post is None:
        import requests
        post = requests.post
    ctx = {
        "args" : args, "template" : template, "sha" : sha, "post" : post,
        "url" : args.base_url.rstrip("/") + "/chat/completions",
        "headers" : {"Authorization" : f"Bearer {key}", "Content-Type" : "application/json"},
        "style" : param_style(args.model, args.param_style),
        "pace" : Pacer(args.min_interval).wait,
    }
    print(f"{args.model}   {ctx['style']} params   target {args.target}   {len(todo)} of {len(rows)} rows to judge   "
          f"workers {args.workers}   min interval {args.min_interval}s   stop after {args.max_consecutive_failures} http failures   prompt sha {sha[:12]}", flush = True)

    def tripped() -> bool:
        if rep.consecutive_http_failures < args.max_consecutive_failures:
            return False
        rep.aborted = True
        print(f"  stopping after {rep.consecutive_http_failures} consecutive http failures. the api is refusing calls. fix the cause and rerun to resume", flush = True)
        return True

    t0 = time.perf_counter()
    dropped : List[str] = []
    first = judge_row(todo[0][1], todo[0][0], ctx, dropped)
    append_jsonl(out, [first])
    tally(rep, first)
    if dropped:
        print(f"  the api refused {' '.join(dropped)} for this model so it is left out from here on", flush = True)

    rest = todo[1 :]
    chunk = max(1, args.workers * 2)
    if not tripped():
        with ThreadPoolExecutor(max_workers = args.workers) as pool:
            for b in range(0, len(rest), chunk):
                part = rest[b : b + chunk]
                results = list(pool.map(lambda ir : judge_row(ir[1], ir[0], ctx, dropped), part))
                append_jsonl(out, results)
                for r in results:
                    tally(rep, r)
                ok = rep.judged - rep.errors
                dt = time.perf_counter() - t0
                print(f"  {rep.skipped + rep.judged}/{len(rows)}   errors {rep.errors}   all pass {rep.all_pass / ok if ok else 0.0:.1%}   "
                      f"{rep.input_tokens + rep.output_tokens} tokens   {dt / 60:.1f} min", flush = True)
                if tripped():
                    break
    rep.seconds = time.perf_counter() - t0
    return rep


def render(rep : JudgeReport, width : int = 76) -> str:
    bar = "=" * width
    ok = rep.judged - rep.errors
    flags = "   ".join(f"{k[6 :]} {v}" for k, v in sorted(rep.flags.items())) or "none"
    return "\n".join([
        bar, "JUDGE", bar,
        f"  out             : {rep.out}",
        f"  judge           : {rep.model}   target {rep.target}",
        f"  rows            : {rep.rows} rows   {rep.skipped} already judged   {rep.judged} judged now   {rep.errors} errors",
        f"  all pass        : {rep.all_pass} of {ok}   {rep.all_pass / ok if ok else 0.0:.1%}",
        f"  constraints     : {rep.n_pass_sum} of {rep.n_constraints_sum} passed   {rep.n_pass_sum / rep.n_constraints_sum if rep.n_constraints_sum else 0.0:.1%}",
        f"  flags           : {flags}",
        f"  tokens          : {rep.input_tokens} in   {rep.output_tokens} out   {rep.seconds:.0f}s",
        f"  stopped early   : after {rep.consecutive_http_failures} consecutive http failures. rerun to resume" if rep.aborted else "  completed       : yes",
        bar,
    ])


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = "score every row of a jsonl with an llm judge driven by a prompt file")
    ap.add_argument("--prompt-file", dest = "prompt_file", help = "rubric with {prompt} {constraints} {response} placeholders")
    ap.add_argument("--rows", help = "jsonl to judge. a generations file or a preference file")
    ap.add_argument("--out", help = "jsonl that receives every input row plus the judge fields")
    ap.add_argument("--model", help = "model name sent to the api")
    ap.add_argument("--target", default = "response", help = "field holding the text to judge. response or chosen or rejected")
    ap.add_argument("--prompt-field", dest = "prompt_field", default = "prompt")
    ap.add_argument("--constraints-field", dest = "constraints_field", default = "constraints")
    ap.add_argument("--sep", default = "|", help = "separator between constraints inside the constraints field")
    ap.add_argument("--limit", type = int, default = 0, help = "first n rows only. 0 means all")
    ap.add_argument("--workers", type = int, default = 8)
    ap.add_argument("--timeout", type = float, default = 120.0)
    ap.add_argument("--retries", type = int, default = 5)
    ap.add_argument("--min-interval", dest = "min_interval", type = float, default = 0.0, help = "seconds between request starts across all workers. 0 means no pacing")
    ap.add_argument("--max-consecutive-failures", dest = "max_consecutive_failures", type = int, default = 8, help = "stop the run after this many rows in a row fail at the http level")
    ap.add_argument("--param-style", dest = "param_style", choices = ("auto", "reasoning", "temperature", "none"), default = "auto")
    ap.add_argument("--reasoning-effort", dest = "reasoning_effort", default = "low")
    ap.add_argument("--temperature", type = float, default = 0.0)
    ap.add_argument("--no-json-mode", dest = "no_json_mode", action = "store_true", help = "do not ask the api for a json object reply")
    ap.add_argument("--base-url", dest = "base_url", default = "https://api.openai.com/v1")
    ap.add_argument("--api-key-env", dest = "api_key_env", default = "OPENAI_API_KEY")
    ap.add_argument("--env-file", dest = "env_file", default = ".env")
    args = ap.parse_args(argv)
    if not (args.prompt_file and args.rows and args.out and args.model):
        raise SystemExit("prompt file and rows and out and model are required")
    rep = run(args)
    print(render(rep))
    return 1 if rep.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
