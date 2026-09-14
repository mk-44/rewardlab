import sys

METHODS = {
    "reward": "rlhf.reward.cli.main",
    "dpo":    "rlhf.dpo.cli.main",
    "ppo":    "rlhf.ppo.cli.main",
    "grpo":   "rlhf.grpo.cli.main",
}

DEFAULT = "reward"


def _load(method):
    from importlib import import_module
    try:
        return import_module(METHODS[method]).main
    except ModuleNotFoundError as e:
        if METHODS[method].split(".")[1] in str(e) or METHODS[method] in str(e):
            print(f"rlhf: '{method}' has no CLI yet", file=sys.stderr)
            raise SystemExit(2)
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in METHODS:
        method, rest = argv[0], argv[1:]
    else:
        method, rest = DEFAULT, argv

    return _load(method)(rest)


if __name__ == "__main__":
    raise SystemExit(main())
