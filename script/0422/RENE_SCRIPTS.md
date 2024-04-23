

MODELS=( "meta-llama/Llama-2-7b-chat-hf" "HuggingFaceH4/zephyr-7b-beta" "berkeley-nest/Starling-LM-7B-alpha" )

v3-D
bash ./script/0422/rene_self-refine-api.sh "api@34.91.163.222:35020/berkeley-nest/Starling-LM-7B-alpha"

# 
bash ./script/0422/rene_self-refine-api.sh "api@104.198.208.112:35020/HuggingFaceH4/zephyr-7b-beta" 


bash ./script/0422/rene_self-refine-api.sh "api@34.170.75.218:35020/meta-llama/Llama-2-7b-chat-hf"
