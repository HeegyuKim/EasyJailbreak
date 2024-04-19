
model=$1
evaluator=$2
defense=$3

if [ -z "$defense" ]; then
    echo "no defense"
    defense=""
fi

python -m experiments.run_jailbreak \
    --target $model \
    --eval $evaluator \
    --defense "$defense" \
    --attacker Jailbroken \
    --limit 1 \
    --prompt_length 1024 \
    --max_new_tokens 128