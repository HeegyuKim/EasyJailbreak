export HF_HOME="/data-plm/hf-home"

# model="api@34.122.186.170:35020/berkeley-nest/Starling-7b-LM-alpha"
# model="api@34.66.3.179:35020/HuggingFaceH4/zephyr-7b-beta"
# model="api@34.122.203.181:35020/meta-llama/Llama-2-7b-chat-hf"
# model="HuggingFaceH4/zephyr-7b-beta"
evaluator=api@34.29.8.219:35020/meta-llama/LlamaGuard-7b

nohup bash script/attack/jailbroken.sh "api@34.122.186.170:35020/berkeley-nest/Starling-7b-LM-alpha" $evaluator > logs/starling.log 2>&1 &
nohup bash script/attack/jailbroken.sh "api@34.66.3.179:35020/HuggingFaceH4/zephyr-7b-beta" $evaluator > logs/zephyr.log 2>&1 &
nohup bash script/attack/jailbroken.sh "api@34.122.203.181:35020/meta-llama/Llama-2-7b-chat-hf" $evaluator > logs/llama.log 2>&1 &
nohup bash script/attack/jailbroken.sh "google/gemma-1.1-7b-it" $evaluator > logs/gemma.log 2>&1 &


# 각 공격기법 PoC
bash script/attack/attack_poc.sh ReNeLLM "google/gemma-1.1-7b-it" api@34.29.8.219:35020/meta-llama/LlamaGuard-7b
bash script/attack/attack_poc.sh GPTFuzz "google/gemma-1.1-7b-it" GPTFuzz


# defense baseline 점검
nohup bash script/attack/attack_defense.sh "google/gemma-1.1-7b-it" api@34.29.8.219:35020/meta-llama/LlamaGuard-7b > logs/gemma.log 2>&1 &



nohup bash script/attack/attack_defense.sh "api@34.122.186.170:35020/berkeley-nest/Starling-7b-LM-alpha" api@34.29.8.219:35020/meta-llama/LlamaGuard-7b > logs/starling.log 2>&1 &
nohup bash script/attack/attack_defense.sh "api@34.66.3.179:35020/HuggingFaceH4/zephyr-7b-beta" api@34.29.8.219:35020/meta-llama/LlamaGuard-7b > logs/zephyr.log 2>&1 &