
attack=$1
model=$2
evaluator=$3
defense=$4

if [ -z "$defense" ]; then
    echo "no defense"
    defense=""
fi

python -m experiments.run_jailbreak \
    --target $model \
    --eval $evaluator \
    --defense "$defense" \
    --attacker "$1" \
    --limit 1 \
    --prompt_length 1024 \
    --max_new_tokens 128