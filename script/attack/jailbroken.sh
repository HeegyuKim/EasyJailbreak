export HF_HOME="/data-plm/hf-home"

python -m run_experiment \
    --target HuggingFaceH4/zephyr-7b-beta \
    --eval meta-llama/LlamaGuard-7b \
    --attacker Jailbroken \
    --prompt_length 1024 \
    --max_new_tokens 128 \