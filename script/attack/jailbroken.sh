export HF_HOME="/data-plm/hf-home"

# model="api@34.122.186.170:35020/berkeley-nest/Starling-7b-LM-alpha"
# model="api@34.66.3.179:35020/HuggingFaceH4/zephyr-7b-beta"
model="HuggingFaceH4/zephyr-7b-beta"
evaluator=api@34.29.8.219:35020/meta-llama/LlamaGuard-7b

python -m experiments.run_jailbreak \
    --target $model \
    --eval $evaluator \
    --attacker Jailbroken \
    --prompt_length 1024 \
    --max_new_tokens 128 \