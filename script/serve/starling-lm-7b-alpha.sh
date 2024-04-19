python -m experiments.run_server \
    --model_name berkeley-nest/Starling-LM-7B-alpha \
    --prompt_length 1024 \
    --max_new_tokens 1024 \
    --fully_sharded_data_parallel False 