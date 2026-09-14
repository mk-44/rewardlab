from __future__ import annotations
import fnmatch
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union
from rlhf.core.contracts import ConfigError


PathLike = Union[str, Path]
TOKEN_VAR = "HF_TOKEN"
HUB_SCHEME = "hf://"
RUNS_ROOT = "experiments"
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
IGNORE_PATTERNS = ("**/.ipynb_checkpoints/**", "**/__pycache__/**", "**/.DS_Store", "**/*.tmp", "hub.json")


@dataclass
class HubPreflight:
    repo_id : str = ""
    user : str = ""
    role : str = ""
    url : str = ""
    private : Optional[bool] = None
    created : bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class HubReport:
    action : str = ""
    repo_id : str = ""
    hub_path : str = ""
    url : str = ""
    commit : str = ""
    message : str = ""
    n_files : int = 0
    n_bytes : int = 0
    seconds : float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def load_dotenv(path : PathLike = ".env", environ : MutableMapping[str, str] = os.environ) -> list:
    p = Path(path)
    if not p.is_file():
        return []

    loaded = []
    for raw in p.read_text(encoding = "utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key or " " in key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1 : -1]
        if key in environ:
            continue
        environ[key] = value
        loaded.append(key)
    return loaded


def resolve_token(environ : Mapping[str, str] = os.environ, dotenv : PathLike = ".env") -> str:
    load_dotenv(dotenv, environ)
    token = environ.get(TOKEN_VAR)
    if token:
        return token
    from huggingface_hub import get_token
    token = get_token()
    if token:
        return token
    raise ConfigError(f"hub: no Hugging Face token. set {TOKEN_VAR} in the environment or in {Path(dotenv).resolve()}, or run `hf auth login`")


def parse_repo_id(repo_id : Any) -> str:
    if not isinstance(repo_id, str) or not _REPO_ID.match(repo_id.strip()):
        raise ConfigError(f"hub.repo_id must look like 'namespace/name', got {repo_id!r}")
    return repo_id.strip()


def parse_hub_uri(uri : str) -> tuple:
    if not isinstance(uri, str) or not uri.startswith(HUB_SCHEME):
        raise ConfigError(f"a hub uri starts with {HUB_SCHEME!r}, got {uri!r}")
    parts = [p for p in uri[len(HUB_SCHEME) :].split("/") if p]
    if len(parts) < 3:
        raise ConfigError(f"hub uri needs 'namespace/name/<run folder>', got {uri!r}")
    return parse_repo_id("/".join(parts[: 2])), "/".join(parts[2 :])


def hub_path_for(out_dir : PathLike, run_name : str, root : PathLike = RUNS_ROOT) -> str:
    run_dir = (Path(out_dir) / run_name).resolve()
    try:
        return run_dir.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return f"{Path(out_dir).name}/{run_name}"


def _api(api : Any, token : Optional[str]):
    if api is not None:
        return api
    from huggingface_hub import HfApi
    return HfApi(token = token or resolve_token())


def preflight(repo_id : str, private : bool = True, api : Any = None, token : Optional[str] = None) -> HubPreflight:
    from huggingface_hub.errors import HfHubHTTPError
    repo_id = parse_repo_id(repo_id)
    api = _api(api, token)
    try:
        who = api.whoami()
    except HfHubHTTPError as e:
        raise ConfigError(f"hub: the token was rejected by the Hub ({e}). check {TOKEN_VAR} or run `hf auth login`")
    user = str(who.get("name", ""))
    role = str(((who.get("auth") or {}).get("accessToken") or {}).get("role", ""))
    if role == "read":
        raise ConfigError(f"hub: the token for {user!r} is read-only; pushing {repo_id} needs a write token")
    existed = bool(api.repo_exists(repo_id))
    try:
        url = str(api.create_repo(repo_id, private = private, exist_ok = True))
    except HfHubHTTPError as e:
        raise ConfigError(f"hub: {user!r} cannot write to {repo_id} ({e}). use a repo under a namespace this token owns")
    info = api.repo_info(repo_id)
    return HubPreflight(
        repo_id = repo_id,
        user = user,
        role = role,
        url = url,
        private = getattr(info, "private", None),
        created = not existed
    )


def _read_json(p : Path) -> dict:
    try:
        return json.loads(p.read_text(encoding = "utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _ignored(rel : str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch("x/" + rel, pat) for pat in IGNORE_PATTERNS)


def run_files(run_dir : PathLike) -> list:
    root = Path(run_dir)
    return sorted(p for p in root.rglob("*") if p.is_file() and not _ignored(p.relative_to(root).as_posix()))


def run_kind(run_dir : PathLike) -> str:
    root = Path(run_dir)
    if (root / "reference.json").exists():
        return "dpo"
    if (root / "model.json").exists():
        return "reward"
    return "run"


def _metrics_line(run_dir : Path) -> str:
    bits = []
    latest = _read_json(run_dir / "checkpoints" / "latest.json")
    best = _read_json(run_dir / "checkpoints" / "best.json")

    if latest.get("step") is not None:
        bits.append(f"step {latest['step']}")
    
    if best.get("step") is not None and best.get("step") != latest.get("step"):
        bits.append(f"best step {best['step']}")
    
    ev = _read_json(run_dir / "eval_best.json") or _read_json(run_dir / "eval_report.json")
    pref = ev.get("preference") or ev.get("overall") or {}
    
    for key, label in (("accuracy", "accuracy"), ("frac_winner_below_ref", "winner_below_ref"), ("mean_margin", "margin")):
        if isinstance(pref.get(key), (int, float)):
            bits.append(f"{label} {pref[key]:.4f}")
    
    gen = ev.get("generation") or {}
    
    if isinstance(gen.get("reward_mean"), (int, float)):
        bits.append(f"gen_reward {gen['reward_mean']:.4f}")
    
    return " ".join(bits)


def commit_message(action : str, run_dir : PathLike, hub_path : str) -> str:
    line = _metrics_line(Path(run_dir))
    return f"{hub_path}: {action}" + (f" {line}" if line else "")


def model_card(run_dir : PathLike, repo_id : str, hub_path : str) -> str:
    root = Path(run_dir)
    kind = run_kind(root)
    best = _read_json(root / "checkpoints" / "best.json")
    latest = _read_json(root / "checkpoints" / "latest.json")
    L = [f"# {hub_path}", ""]
    L.append(f"A rewardlab {kind} run, mirrored from its run directory. Every file the run wrote is here: "
             f"config stamps, metrics.jsonl, console.log, the checkpoints with their pointers, and the eval reports.")
    L.append("")
    if best or latest:
        L.append("## Checkpoints")
        L.append("")
        if best:
            L.append(f"* best: `checkpoints/{best.get('file', '?')}` at step {best.get('step', '?')}")
        if latest:
            L.append(f"* latest: `checkpoints/{latest.get('file', '?')}` at step {latest.get('step', '?')}")
        L.append("")
    line = _metrics_line(root)
    if line:
        L += ["## Metrics", "", line, ""]
    L += ["## Use", "", "```", f"python -m rlhf {kind if kind != 'run' else 'dpo'} pull --hf-repo-id {repo_id} --config <the run's yaml>", "```", ""]
    if kind == "reward":
        L += ["```python", "from rlhf.reward.scoring import from_run", f"scorer = from_run(\"{HUB_SCHEME}{repo_id}/{hub_path}\")", "```", ""]
    if kind == "dpo":
        L += ["```python", "from transformers import AutoModelForCausalLM",
              f"model = AutoModelForCausalLM.from_pretrained(\"{repo_id}\", subfolder = \"{hub_path}/export\")", "```", ""]
    L += ["## Files", ""]
    for p in run_files(root):
        L.append(f"* `{p.relative_to(root).as_posix()}`")
    L.append("")
    return "\n".join(L)


def push_run(
    run_dir : PathLike,
    repo_id : str,
    hub_path : str,
    message : Optional[str] = None,
    action : str = "push",
    api : Any = None,
    token : Optional[str] = None,
) -> HubReport:
    root = Path(run_dir)
    if not root.is_dir():
        raise ConfigError(f"hub: run directory not found: {root}")
    repo_id = parse_repo_id(repo_id)
    hub_path = hub_path.strip("/")
    if not hub_path:
        raise ConfigError("hub: hub_path is empty; a run needs its own folder in the repo")
    api = _api(api, token)
    (root / "README.md").write_text(model_card(root, repo_id, hub_path), encoding = "utf-8")
    files = run_files(root)
    message = message or commit_message(action, root, hub_path)
    t0 = time.perf_counter()
    info = api.upload_folder(
        repo_id = repo_id,
        folder_path = str(root),
        path_in_repo = hub_path,
        commit_message = message,
        ignore_patterns = list(IGNORE_PATTERNS),
        delete_patterns = ["checkpoints/*"],
    )
    rep = HubReport(
        action = action,
        repo_id = repo_id,
        hub_path = hub_path,
        url = str(getattr(info, "commit_url", None) or getattr(info, "repo_url", None) or info),
        commit = str(getattr(info, "oid", "") or ""),
        message = message,
        n_files = len(files),
        n_bytes = sum(p.stat().st_size for p in files),
        seconds = time.perf_counter() - t0,
    )
    (root / "hub.json").write_text(json.dumps(rep.to_dict(), indent = 2) + "\n", encoding = "utf-8")
    return rep


def pull_run(repo_id : str, hub_path : str, dest : PathLike, api : Any = None, token : Optional[str] = None) -> HubReport:
    repo_id = parse_repo_id(repo_id)
    hub_path = hub_path.strip("/")
    if not hub_path:
        raise ConfigError("hub: hub_path is empty. say which run folder to pull")
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise ConfigError(f"hub: {dest} exists and is not empty. pull into a fresh run directory")
    api = _api(api, token)
    
    tmp = dest.parent / f".{dest.name}.pull"
    shutil.rmtree(tmp, ignore_errors = True)
    tmp.mkdir(parents = True)
    t0 = time.perf_counter()
    
    try:
        api.snapshot_download(repo_id = repo_id, allow_patterns = [f"{hub_path}/*"], local_dir = str(tmp))
        src = tmp / hub_path
        if not src.is_dir():
            raise ConfigError(f"hub: {repo_id} has no folder {hub_path!r}")
        dest.parent.mkdir(parents = True, exist_ok = True)
        if dest.exists():
            dest.rmdir()
        shutil.move(str(src), str(dest))
    finally:
        shutil.rmtree(tmp, ignore_errors = True)
    files = run_files(dest)
    rep = HubReport(
        action = "pull",
        repo_id = repo_id,
        hub_path = hub_path,
        url = f"https://huggingface.co/{repo_id}/tree/main/{hub_path}",
        n_files = len(files),
        n_bytes = sum(p.stat().st_size for p in files),
        seconds = time.perf_counter() - t0,
    )
    (dest / "hub.json").write_text(json.dumps(rep.to_dict(), indent = 2) + "\n", encoding = "utf-8")
    return rep


def render_preflight(p : HubPreflight, width : int = 76) -> str:
    bar = "=" * width
    vis = "private" if p.private else ("public" if p.private is False else "unknown visibility")
    return "\n".join([
        bar, "HUB", bar,
        f"  repo          : {p.repo_id}   ({'created' if p.created else 'exists'}, {vis})",
        f"  token         : {p.user}   role {p.role or '?'}",
        f"  url           : {p.url}",
        bar,
    ])


def render(rep : HubReport, width : int = 76) -> str:
    bar = "=" * width
    return "\n".join([
        bar, f"HUB {rep.action.upper()}", bar,
        f"  repo          : {rep.repo_id} / {rep.hub_path}",
        f"  files         : {rep.n_files:,}   {rep.n_bytes / 1e6:,.1f} MB   {rep.seconds:.1f} s",
        f"  commit        : {rep.commit or '-'}",
        f"  message       : {rep.message}" if rep.message else f"  url           : {rep.url}",
        bar,
    ])
