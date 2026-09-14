#!/bin/sh
set -eu
cd "$(dirname "$0")/../../.."

CFG=experiments/dpo/non_sft_50k_data/config.yaml
RM_CKPT_DIR=experiments/reward/rhyme_50k_distilbert/20260905_025233_e4/checkpoints

for f in best.json step_00006500.pt; do
    if [ ! -f "$RM_CKPT_DIR/$f" ]; then
        echo "missing $RM_CKPT_DIR/$f: copy best.json and step_00006500.pt from the reward run there first" >&2
        exit 2
    fi
done

python -m rlhf dpo build-cache --config "$CFG"
python -m rlhf dpo train       --config "$CFG" "$@"
python -m rlhf dpo eval        --config "$CFG" --which best
python -m rlhf dpo export      --config "$CFG" --which best
