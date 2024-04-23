# python -m experiments.run_autodan --model_name berkeley-nest/Starling-LM-7B-alpha --template openchat_3.5
# python -m experiments.run_autodan --model_name HuggingFaceH4/zephyr-7b-beta --template zephyr
# python -m experiments.run_autodan --model_name meta-llama/Llama-2-7b-chat-hf --template llama-2


# MODELS=( "meta-llama/Llama-2-7b-chat-hf" "HuggingFaceH4/zephyr-7b-beta" "berkeley-nest/Starling-LM-7B-alpha" )
DEFENSES=( "self-reminder" "in-context" "smooth-llm" )

for defense in ${DEFENSES[@]}; do
    python -m experiments.run_autodan --model_name berkeley-nest/Starling-LM-7B-alpha --template openchat_3.5 --defense $defense
done
