
evaluator=GPTFuzz


attack(){
    model=$1
    defense=$2

    if [ -z "$defense" ]; then
        echo "no defense"
        defense=""
    fi

    python -m experiments.run_jailbreak \
        --target $model \
        --eval $evaluator \
        --defense "$defense" \
        --attacker GPTFuzz \
        --limit 100 \
        --prompt_length 1024 \
        --max_new_tokens 128
}

if [ $1 -eq 0 ]; then
    MODELS=( "meta-llama/Llama-2-7b-chat-hf" )
else
    # MODELS=( "HuggingFaceH4/zephyr-7b-beta" "berkeley-nest/Starling-LM-7B-alpha" )
    MODELS=( "berkeley-nest/Starling-LM-7B-alpha" )
fi

DEFENSES=( "self-reminder" "in-context" "smooth-llm" )

for model in ${MODELS[@]}; do
    attack $model
    for defense in ${DEFENSES[@]}; do
        attack $model $defense
    done
done

# attack 
# attack "self-reminder"
# attack "in-context"
# attack "smooth-llm"