# non_sft_50k_data

Direct DPO on base `gpt2` over the `rhyme_50k` pairs, no SFT leg. Every command runs from the repo root.

1. `git clone https://github.com/mk-44/rewardlab.git && cd rewardlab`
2. `pip install -e ".[rhyme]"`
3. Copy `best.json` and `step_00006500.pt` from the reward run into `experiments/reward/rhyme_50k_distilbert/20260905_025233_e4/checkpoints/`
4. `sh experiments/dpo/non_sft_50k_data/run.sh`
5. Interrupted? `sh experiments/dpo/non_sft_50k_data/run.sh --resume`
6. To mirror the run to the Hugging Face Hub put `HF_TOKEN=<token>` in `.env` at the repo root and set `hub.repo_id: <namespace>/<name>` in `config.yaml`. Every verb then pushes the run directory when it finishes. `python -m rlhf dpo push --config experiments/dpo/non_sft_50k_data/config.yaml` pushes a finished run by hand and `pull` fetches one into an empty run directory. `--hf-repo-id <namespace>/<name>` on any verb is the same as setting `hub.repo_id`.

Outputs land in `experiments/dpo/non_sft_50k_data/run/`: `metrics.jsonl`, `console.log`, `checkpoints/`, `eval_report.json`, `export/`.

The device is `execution.device` in `config.yaml`, set to `cuda`; change it there for another machine. The reward model scores on `RM_DEVICE` in `custom_reward_functions.py`, `cpu` unless you set it. Both reward functions live in that file; `config.yaml` names them as `<file>.py:<function>`.
