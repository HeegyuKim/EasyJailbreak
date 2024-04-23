model=$1
evaluator=api@34.29.8.219:35020/meta-llama/LlamaGuard-7b
defense="self-refine"

python -m experiments.run_jailbreak \
    --target $model \
    --eval $evaluator \
    --defense "$defense" \
    --attacker ReNeLLM \
    --limit 100 \
    --prompt_length 1024 \
    --max_new_tokens 1024