export HF_HOME="/data-plm/hf-home"

# llama defense 다시 돌리기
nohup bash script/attack/attack_defense.sh "api@34.170.75.218:35020/meta-llama/Llama-2-7b-chat-hf" api@34.29.8.219:35020/meta-llama/LlamaGuard-7b > logs/llama-real.log 2>&1 &

# ReNeLLM 공격 실험
bash script/attack/attack_poc.sh ReNeLLM "meta-llama/Llama-2-7b-chat-hf" api@34.29.8.219:35020/meta-llama/LlamaGuard-7b
bash script/attack/attack_poc.sh GPTFuzz "meta-llama/Llama-2-7b-chat-hf" GPTFuzz

# ReNeLLM 돌려보자
bash script/attack/attack.sh ReNeLLM "meta-llama/Llama-2-7b-chat-hf" api@34.29.8.219:35020/meta-llama/LlamaGuard-7b

# Self-Refine
