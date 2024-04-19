
model=$1
evaluator=$2

bash script/attack/jailbroken.sh $model $evaluator self-reminder
bash script/attack/jailbroken.sh $model $evaluator in-context
bash script/attack/jailbroken.sh $model $evaluator smooth-llm