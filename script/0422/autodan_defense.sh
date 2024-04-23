
MODELS=( "meta-llama/Llama-2-7b-chat-hf" "HuggingFaceH4/zephyr-7b-beta" "berkeley-nest/Starling-LM-7B-alpha" )
# DEFENSES=( "self-reminder" "in-context" "smooth-llm" )
DEFENSES=( "self-refine-adv-v1" )
for model in ${MODELS[@]}; do
    for defense in ${DEFENSES[@]}; do
        python -m experiments.defense_autodan --model_name $model --defense $defense
    done
done
