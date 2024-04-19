
model=$1
evaluator=$2

python -m experiments.run_jailbreak \
    --target $model \
    --eval $evaluator \
    --attacker Jailbroken \
    --limit 100 \
    --prompt_length 1024 \
    --max_new_tokens 128