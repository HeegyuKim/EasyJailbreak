
model=$1
evaluator=api@34.29.8.219:35020/meta-llama/LlamaGuard-7b

if [ -z "$model" ]; then
    echo "no model"
    exit 1
fi

# api@34.122.186.170:35020/berkeley-nest/Starling-LM-7B-alpha
# api@34.66.3.179:35020/HuggingFaceH4/zephyr-7b-beta
# api@34.122.203.181:35020/meta-llama/Llama-2-7b-chat-hf


attack(){
    defense=$1

    if [ -z "$defense" ]; then
        echo "no defense"
        defense=""
    fi

    python -m experiments.run_jailbreak \
        --target $model \
        --eval $evaluator \
        --defense "$defense" \
        --attacker ReNeLLM \
        --limit 100 \
        --prompt_length 1024 \
        --max_new_tokens 1024
}
attack 
attack "self-reminder"
attack "in-context"
attack "smooth-llm"