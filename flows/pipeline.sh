#!/usr/bin/env bash
# Full command-level agent pipeline (docs/plan.md, steps 3-5). Run from the repo root.
# Logs: results/agent/pipeline.log; every episode: results/agent/agent_runs.jsonl
# FROM=eval_qwen (or train, eval_lora) resumes at that stage.
set -e
TRAIN="mac16 mac12 fir4 alu32"
HELD="mac20 simd4"
FROM=${FROM:-prep}
stages="prep eval_random eval_qwen train eval_lora"
run=0
for s in $stages; do
    [ "$s" = "$FROM" ] && run=1
    [ $run = 1 ] || continue
    echo "=== stage $s $(date)"
    case $s in
        prep)        python -u -W ignore -m flows.agent prep mac12 fir4 alu32 mac20 simd4 ;;
        eval_random) python -u -W ignore -m flows.agent eval --designs $TRAIN $HELD --policies random --seeds 2 ;;
        eval_qwen)   python -u -W ignore -m flows.agent eval --designs $TRAIN $HELD --policies qwen --seeds 2 ;;
        train)       python -u -W ignore -m flows.agent train --designs $TRAIN --episodes 24 ;;
        eval_lora)   python -u -W ignore -m flows.agent eval --designs $TRAIN $HELD --policies lora --seeds 2 ;;
    esac
done
echo "=== done $(date)"
