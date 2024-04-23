MODELS=( "meta-llama/Llama-2-7b-chat-hf" ) # "HuggingFaceH4/zephyr-7b-beta" "berkeley-nest/Starling-LM-7B-alpha" )
DEFENSES=( "self-reminder" "in-context" "smooth-llm" )

for model in ${MODELS[@]}; do

    for defense in ${DEFENSES[@]}; do
        python -m experiments.mt_judge --model "$model-$defense" --filename "MT-Bench-$defense.jsonl"
    done
done
