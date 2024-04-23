# ./script/0423/refine_server.sh api@35.204.152.13:35020/meta-llama/Llama-2-7b-chat-hf
# ./script/0423/refine_server.sh api@34.91.163.222:35020/berkeley-nest/Starling-LM-7B-alpha
model=$1
python -m experiments.refine --model $model --defense "self-refine"
python -m experiments.refine --model $model --defense "self-refine-adv-v1"