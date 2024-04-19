export HF_HOME="/data-plm/hf-home"

# Felladrin/TinyMistral-248M-Chat-v2
python -m experiments.run_server \
    --model_name meta-llama/LlamaGuard-7b \
    --prompt_length 1024 \
    --max_new_tokens 128