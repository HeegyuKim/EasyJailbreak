

# MODELS=( "meta-llama/Llama-2-7b-chat-hf" "HuggingFaceH4/zephyr-7b-beta" "berkeley-nest/Starling-LM-7B-alpha" )
MODELS=( "berkeley-nest/Starling-LM-7B-alpha" )
DEFENSES=( "self-reminder" "in-context" "smooth-llm" )

for model in ${MODELS[@]}; do
    python -m instruction_following_eval.evaluation_main \
        --input_data instruction_following_eval/data/input_data.jsonl \
        --input_response_data "outputs/$model/IFEval.jsonl" \
        --output_dir "outputs/$model/IFEval/no-defense/"

    # for defense in ${DEFENSES[@]}; do
    #     python -m instruction_following_eval.evaluation_main \
    #         --input_data instruction_following_eval/data/input_data.jsonl \
    #         --input_response_data "outputs/$model/IFEval-$defense.jsonl" \
    #         --output_dir "outputs/$model/IFEval/$defense/"
    # done
done
