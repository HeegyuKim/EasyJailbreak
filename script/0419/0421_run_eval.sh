export HF_HOME="/data-plm/hf-home"


# MODELS=( "meta-llama/Llama-2-7b-chat-hf" "HuggingFaceH4/zephyr-7b-beta" "berkeley-nest/Starling-7b-LM-alpha" )
MODELS=( "berkeley-nest/Starling-LM-7B-alpha" )
for model in ${MODELS[@]}; do
    python -m fastchat.llm_judge.gen_flax_api_answer --model $model 
    bash script/eval/ifeval.sh "$model"
done
