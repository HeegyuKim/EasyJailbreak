export HF_HOME="/data-plm/hf-home"

# model="api@34.122.186.170:35020/berkeley-nest/Starling-7b-LM-alpha"
model="34.66.3.179:35020/HuggingFaceH4/zephyr-7b-beta"

python -m fastchat.llm_judge.gen_flax_api_answer --model $model 