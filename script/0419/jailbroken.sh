export HF_HOME="/data-plm/hf-home"

# model="api@34.122.186.170:35020/berkeley-nest/Starling-7b-LM-alpha"
# model="api@34.66.3.179:35020/HuggingFaceH4/zephyr-7b-beta"
# model="api@34.122.203.181:35020/meta-llama/Llama-2-7b-chat-hf"
# model="HuggingFaceH4/zephyr-7b-beta"
evaluator=api@34.29.8.219:35020/meta-llama/LlamaGuard-7b

nohup bash script/attack/jailbroken.sh "api@34.122.186.170:35020/berkeley-nest/Starling-7b-LM-alpha" $evaluator > logs/starling.log 2>&1 &
nohup bash script/attack/jailbroken.sh "api@34.66.3.179:35020/HuggingFaceH4/zephyr-7b-beta" $evaluator > logs/zephyr.log 2>&1 &
nohup bash script/attack/jailbroken.sh "api@34.122.203.181:35020/meta-llama/Llama-2-7b-chat-hf" $evaluator > logs/llama.log 2>&1 &
