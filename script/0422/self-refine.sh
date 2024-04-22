export HF_HOME="/data/hf-home"
export LD_LIBRARY_PATH=~/anaconda3/envs/easydel/lib/


MODELS=( "meta-llama/Llama-2-7b-chat-hf" "HuggingFaceH4/zephyr-7b-beta" "berkeley-nest/Starling-LM-7B-alpha" )
# MODELS=( "berkeley-nest/Starling-LM-7B-alpha" )
for model in ${MODELS[@]}; do
    python -m experiments.refine --model $model 
done
