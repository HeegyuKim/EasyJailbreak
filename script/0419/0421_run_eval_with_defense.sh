export HF_HOME="/data-plm/hf-home"


MODELS=( "meta-llama/Llama-2-7b-chat-hf" "HuggingFaceH4/zephyr-7b-beta")
# MODELS=( "HuggingFaceH4/zephyr-7b-beta" )
# MODELS=( "api@34.91.163.222:35020/berkeley-nest/Starling-LM-7B-alpha" )
DEFENSES=( "self-reminder" "in-context" "smooth-llm" )

for model in ${MODELS[@]}; do
    for defense in ${DEFENSES[@]}; do
        python -m fastchat.llm_judge.gen_flax_api_answer \
            --model $model \
            --defense $defense
        
        python -m instruction_following_eval.generate_main \
            --model $model \
            --defense $defense
    done
done
